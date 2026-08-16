
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import gc
from tqdm import tqdm
import random
import argparse
import sys
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, average_precision_score
from thop import profile, clever_format


# ===================== 1. 配置 =====================
class GlobalConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    batch_size = 32
    lr = 0.0001

    # 默认值，可通过命令行参数覆盖
    healthy_root = "./Data/TeaFold4/healthy_age_classify"
    disease_root = "./Data/TeaFold4/teaLeafBD"

    grade_classes = ["T1", "T2", "T3", "T4"]
    disease_classes = ["1. Tea algal leaf spot", "2. Brown Blight", "3. Gray Blight",
                       "4. Helopeltis", "5. Red spider", "6. Green mirid bug"]


cfg = GlobalConfig()


# ===================== 2. 数据集 =====================
class TeaMultiTaskDataset(Dataset):
    def __init__(self, split_type='train', transform=None):
        self.samples = []
        self.transform = transform

        # 健康茶叶数据
        h_path = os.path.join(cfg.healthy_root, split_type)
        if os.path.exists(h_path):
            for idx, g_name in enumerate(cfg.grade_classes):
                folder = os.path.join(h_path, g_name)
                if os.path.exists(folder):
                    for img in os.listdir(folder):
                        self.samples.append((os.path.join(folder, img), 0, idx, -1))

        # 病害数据
        temp_dis = []
        for idx, d_name in enumerate(cfg.disease_classes):
            folder = os.path.join(cfg.disease_root, d_name)
            if os.path.exists(folder):
                for img in os.listdir(folder):
                    temp_dis.append((os.path.join(folder, img), 1, -1, idx))

        if temp_dis:
            np.random.shuffle(temp_dis)
            n = len(temp_dis)
            if split_type == 'train':
                self.samples.extend(temp_dis[:int(n * 0.7)])
            elif split_type == 'val':
                self.samples.extend(temp_dis[int(n * 0.7):int(n * 0.85)])
            else:
                self.samples.extend(temp_dis[int(n * 0.85):])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, b, g, d = self.samples[i]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, torch.tensor(b), torch.tensor(g), torch.tensor(d)
        except:
            return torch.zeros(3, 224, 224), torch.tensor(0), torch.tensor(-1), torch.tensor(-1)


# ===================== 3. 改进版自融合网络 =====================
class FusionNet_SelfBilinear_v2(nn.Module):
    """
    改进版自双线性融合：一阶全局特征 (GAP) + 二阶纹理特征 (Bilinear)
    平衡病害分类与等级分类的需求。
    """

    def __init__(self, mobilenet_feat, reduce_dim=128):
        super().__init__()
        self.mobilenet = mobilenet_feat

        # --- 分支 1：一阶特征 (捕捉宏观病害斑块、结构) ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        # MobileNetV3 small 的默认通道数是 576

        # --- 分支 2：二阶特征 (捕捉微观细粒度纹理、等级差异) ---
        self.reduce_conv = nn.Conv2d(576, reduce_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduce_dim)
        self.relu1 = nn.ReLU(inplace=True)

        bilinear_dim = reduce_dim * reduce_dim  # 128*128 = 16384

        # 将庞大的双线性特征压缩，防止参数量掩盖了GAP特征
        bilinear_dim2 = 256
        self.compress_bili = nn.Linear(bilinear_dim, bilinear_dim2)
        self.bn2 = nn.BatchNorm1d(bilinear_dim2)
        self.relu2 = nn.ReLU(inplace=True)

        # --- 特征融合头 ---
        # 576 (GAP) + 256 (Bilinear compressed) = 832
        combined_dim = 576 + bilinear_dim2
        self.dropout = nn.Dropout(0.4)

        self.fc_bin = nn.Linear(combined_dim, 2)
        self.fc_gra = nn.Linear(combined_dim, 4)
        self.fc_dis = nn.Linear(combined_dim, 6)

    def forward(self, x):
        # 提取空间特征图 [B, 576, H, W]
        feat_map = self.mobilenet(x)

        # --- 提取分支 1 (GAP) ---
        gap_feat = self.gap(feat_map).view(x.size(0), -1)  # [B, 576]

        # --- 提取分支 2 (Bilinear) ---
        reduced_feat = self.relu1(self.bn1(self.reduce_conv(feat_map)))  # [B, 128, H, W]
        B, C, H, W = reduced_feat.size()
        reduced_feat = reduced_feat.view(B, C, H * W)

        # 外积计算协方差
        bilinear_feat = torch.bmm(reduced_feat, reduced_feat.transpose(1, 2)) / (H * W)
        bilinear_feat = bilinear_feat.view(B, -1)  # [B, 16384]

        # 归一化 (Signed Sqrt + L2)
        bilinear_feat = torch.sign(bilinear_feat) * torch.sqrt(torch.abs(bilinear_feat) + 1e-8)
        bilinear_feat = torch.nn.functional.normalize(bilinear_feat, dim=1)

        # 压缩二阶特征
        bili_compressed = self.relu2(self.bn2(self.compress_bili(bilinear_feat)))  # [B, 256]

        # --- 融合 (拼接) ---
        fused_feat = torch.cat([gap_feat, bili_compressed], dim=1)  # [B, 832]
        fused_feat = self.dropout(fused_feat)

        return self.fc_bin(fused_feat), self.fc_gra(fused_feat), self.fc_dis(fused_feat)


# ===================== 4. 模型构建 =====================
def build_model(name):
    base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    mob_feat = base.features

    if name == 'GBDFNet':
        return FusionNet_SelfBilinear_v2(mob_feat, reduce_dim=128)
    return None


# ===================== 5. 工具函数 =====================
def macro_accuracy(true_labels, pred_labels):
    classes = np.unique(true_labels)
    class_accuracies = []
    for cls in classes:
        cls_true = (true_labels == cls)
        cls_pred = (pred_labels == cls)
        correct = np.sum(cls_true & cls_pred)
        total = np.sum(cls_true)
        class_accuracies.append(0.0 if total == 0 else correct / total)
    return np.mean(class_accuracies)


def calculate_map(true_labels, pred_probs, num_classes):
    """
    计算 mAP (mean Average Precision)
    """
    true_labels_onehot = np.zeros((len(true_labels), num_classes))
    for i, label in enumerate(true_labels):
        true_labels_onehot[i, label] = 1

    aps = []
    for i in range(num_classes):
        ap = average_precision_score(true_labels_onehot[:, i], pred_probs[:, i])
        aps.append(ap)

    map_value = np.mean(aps)
    return map_value, aps


def print_metrics(true_g, pred_g, prob_g, true_d, pred_d, prob_d):
    print("\n========== 健康分级 (Grade) 指标 ==========")
    if len(true_g) > 0:
        cm_g = confusion_matrix(true_g, pred_g)
        print("混淆矩阵:\n", cm_g)
        class_acc_g = cm_g.diagonal() / cm_g.sum(axis=1)
        for i, cls in enumerate(cfg.grade_classes):
            print(f"{cls} 准确率: {class_acc_g[i]:.2%}")

        f1_scores = f1_score(true_g, pred_g, average=None)
        print(f"每类F1分数: {[f'{f:.4f}' for f in f1_scores]}")
        print(f"宏平均F1分数: {f1_score(true_g, pred_g, average='macro'):.4f}")
        print(f"宏准确率（Macro Acc）: {macro_accuracy(true_g, pred_g):.4f}")
        print(f"整体准确率（Overall Acc）: {accuracy_score(true_g, pred_g):.4f}")

        # 计算 mAP
        map_g, aps_g = calculate_map(true_g, prob_g, len(cfg.grade_classes))
        print(f"每类AP: {[f'{ap:.4f}' for ap in aps_g]}")
        print(f"平均精度均值 (mAP): {map_g:.4f}")

    print("\n========== 病害分类 (Disease) 指标 ==========")
    if len(true_d) > 0:
        cm_d = confusion_matrix(true_d, pred_d)
        print("混淆矩阵:\n", cm_d)
        class_acc_d = cm_d.diagonal() / cm_d.sum(axis=1)
        for i, cls in enumerate(cfg.disease_classes):
            print(f"{cls} 准确率: {class_acc_d[i]:.2%}")

        f1_scores = f1_score(true_d, pred_d, average=None)
        print(f"每类F1分数: {[f'{f:.4f}' for f in f1_scores]}")
        print(f"宏平均F1分数: {f1_score(true_d, pred_d, average='macro'):.4f}")
        print(f"宏准确率（Macro Acc）: {macro_accuracy(true_d, pred_d):.4f}")
        print(f"整体准确率（Overall Acc）: {accuracy_score(true_d, pred_d):.4f}")

        # 计算 mAP
        map_d, aps_d = calculate_map(true_d, prob_d, len(cfg.disease_classes))
        print(f"每类AP: {[f'{ap:.4f}' for ap in aps_d]}")
        print(f"平均精度均值 (mAP): {map_d:.4f}")


# ===================== 6. 训练逻辑 =====================
def train_and_eval(name, loaders):
    print(f"\n🚀 开始训练: {name} ")

    # 记录模型开始时间
    model_start_time = time.time()

    model = build_model(name)
    if not model:
        print(f"模型 {name} 构建失败")
        return None
    model.to(cfg.device)

    dummy = torch.randn(1, 3, 224, 224).to(cfg.device)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    f_s, p_s = clever_format([flops, params], "%.3f")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best_score = -1.0
    best_state = None

    train_loader, val_loader, test_loader = loaders

    for epoch in range(cfg.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False)
        for imgs, b, g, d in loop:
            imgs, b, g, d = imgs.to(cfg.device), b.to(cfg.device), g.to(cfg.device), d.to(cfg.device)
            opt.zero_grad()
            pb, pg, pd = model(imgs)

            loss_b = ce(pb, b)
            loss_g = ce(pg[b == 0], g[b == 0]) if (b == 0).any() else 0.0
            loss_d = ce(pd[b == 1], d[b == 1]) if (b == 1).any() else 0.0

            loss = 0.2 * loss_b + 0.4 * loss_g + 0.4 * loss_d
            if isinstance(loss, torch.Tensor):
                loss.backward()
                opt.step()
                loop.set_postfix(loss=loss.item())

        model.eval()
        correct_g, total_g = 0, 1e-6
        correct_d, total_d = 0, 1e-6
        with torch.no_grad():
            for imgs, b, g, d in val_loader:
                imgs = imgs.to(cfg.device)
                b, g, d = b.to(cfg.device), g.to(cfg.device), d.to(cfg.device)
                _, pg, pd = model(imgs)

                mask_g = (b == 0)
                if mask_g.any():
                    correct_g += (pg[mask_g].argmax(1) == g[mask_g]).sum().item()
                    total_g += mask_g.sum().item()

                mask_d = (b == 1)
                if mask_d.any():
                    correct_d += (pd[mask_d].argmax(1) == d[mask_d]).sum().item()
                    total_d += mask_d.sum().item()

        acc_g = correct_g / total_g
        acc_d = correct_d / total_d
        score = (acc_g + acc_d) / 2.0
        print(f"   Val -> Grade: {acc_g:.2%} | Disease: {acc_d:.2%} | Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_state = model.state_dict()

    if best_state: model.load_state_dict(best_state)
    model.eval()

    t_stats = {"gc": 0, "gt": 0, "dc": 0, "dt": 0}
    all_true_g, all_pred_g, all_prob_g = [], [], []
    all_true_d, all_pred_d, all_prob_d = [], [], []

    with torch.no_grad():
        for imgs, b, g, d in test_loader:
            imgs = imgs.to(cfg.device)
            b, g, d = b.to(cfg.device), g.to(cfg.device), d.to(cfg.device)
            _, pg, pd = model(imgs)

            mask_g = (b == 0)
            if mask_g.any():
                all_true_g.extend(g[mask_g].cpu().numpy())
                all_pred_g.extend(pg[mask_g].argmax(1).cpu().numpy())
                all_prob_g.extend(torch.softmax(pg[mask_g], dim=1).cpu().numpy())  # 收集概率
                t_stats["gc"] += (pg[mask_g].argmax(1) == g[mask_g]).sum().item()
                t_stats["gt"] += mask_g.sum().item()

            mask_d = (b == 1)
            if mask_d.any():
                all_true_d.extend(d[mask_d].cpu().numpy())
                all_pred_d.extend(pd[mask_d].argmax(1).cpu().numpy())
                all_prob_d.extend(torch.softmax(pd[mask_d], dim=1).cpu().numpy())  # 收集概率
                t_stats["dc"] += (pd[mask_d].argmax(1) == d[mask_d]).sum().item()
                t_stats["dt"] += mask_d.sum().item()

    final_acc_g = t_stats["gc"] / (t_stats["gt"] + 1e-6)
    final_acc_d = t_stats["dc"] / (t_stats["dt"] + 1e-6)

    # 传入概率数据以计算 mAP
    print_metrics(
        np.array(all_true_g), np.array(all_pred_g), np.array(all_prob_g),
        np.array(all_true_d), np.array(all_pred_d), np.array(all_prob_d)
    )

    # 计算并输出模型运行时间
    model_end_time = time.time()
    model_duration = model_end_time - model_start_time
    print(f"⏱️ 模型 {name} 实际运行时间: {model_duration:.2f} 秒 ({model_duration / 60:.2f} 分钟)")
    print(f"参数量: {p_s}")
    print(f"FLOPs: {f_s}")

    return {
        "模型": name, "参数量": p_s, "FLOPs": f_s,
        "运行时间(秒)": round(model_duration, 2),
        "健康分级Acc": round(final_acc_g, 4),
        "病害分类Acc": round(final_acc_d, 4)
    }


# ===================== 7. 主函数 =====================
def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="茶叶叶片分类模型训练与评估脚本")
    parser.add_argument('--healthy_root', type=str, default="./Data/TeaFold4/healthy_age_classify",
                        help='健康叶片数据根目录路径')
    parser.add_argument('--disease_root', type=str, default="./Data/TeaFold4/teaLeafBD", help='病害叶片数据根目录路径')
    args = parser.parse_args()

    # 更新配置
    cfg.healthy_root = args.healthy_root
    cfg.disease_root = args.disease_root

    output_file = "outputFold4.txt"

    # 重定向标准输出到文件和控制台
    class TeeOutput:
        def __init__(self, file_obj, original_stdout):
            self.file = file_obj
            self.original_stdout = original_stdout

        def write(self, message):
            self.file.write(message)
            self.original_stdout.write(message)

        def flush(self):
            self.file.flush()
            self.original_stdout.flush()

    # 使用 'w' 模式覆盖旧文件，确保每次运行都是新结果
    with open(output_file, 'a', encoding='utf-8') as f:
        tee = TeeOutput(f, sys.stdout)
        old_stdout = sys.stdout
        sys.stdout = tee

        try:
            start = time.time()

            seed = 42
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            print("加载数据...")
            print(f"Healthy root: {cfg.healthy_root}")
            print(f"Disease root: {cfg.disease_root}")

            train_ds = TeaMultiTaskDataset('train', tf)
            val_ds = TeaMultiTaskDataset('val', tf)
            test_ds = TeaMultiTaskDataset('test', tf)

            print(f"\n📊 数据集统计 -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

            loaders = [
                DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0),
                DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0),
                DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
            ]

            models_to_run = ['GBDFNet']
            results = []

            for m in models_to_run:
                try:
                    res = train_and_eval(m, loaders)
                    if res:
                        results.append(res)
                except Exception as e:
                    print(f"❌ {m} 报错: {e}")
                    import traceback
                    traceback.print_exc()
                gc.collect()

            # 输出所有模型的运行时间汇总
            if results:
                print("\n" + "=" * 30)
                print("⏱️ 模型运行时间汇总:")
                print("=" * 30)
                for res in results:
                    if "运行时间(秒)" in res:
                        print(f"模型: {res['模型']:<25} | 耗时: {res['运行时间(秒)']:>8.2f} 秒")
                print("=" * 30)

            print(f"\n全部完成，总耗时 {(time.time() - start) / 60:.1f} 分钟")
            print(f"详细结果已输出到 {output_file}")

        finally:
            # 恢复标准输出
            sys.stdout = old_stdout


if __name__ == "__main__":
    main()

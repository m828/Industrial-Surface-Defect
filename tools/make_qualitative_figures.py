#!/usr/bin/env python3
"""
定性可视化：筛选典型样本生成对比图。

典型样本类别：
1. 小目标缺陷
2. 细长划痕
3. 多缺陷样本
4. 背景纹理干扰强
5. Base-B 漏检但 A2MS-DefectNet-B 检出
6. Base-B 边界断裂但 A2MS-DefectNet-B 更完整

输出目录：figures/qualitative_results/
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import importlib.util
from scipy import ndimage
from torch.utils import data
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

WORKSPACE = "/workspace/Industrial Surface Defect"
NEW_DIR = os.path.join(WORKSPACE, "new")
NEW_COPY_DIR = os.path.join(WORKSPACE, "new (copy)")

sys.path.insert(0, NEW_DIR)
sys.path.insert(0, NEW_COPY_DIR)

# ============================================================
# 类别定义
# ============================================================
LEATHER_CLASSES = {
    0: "background", 1: "open_wound", 2: "scratch", 3: "brand_mark",
    4: "hole", 5: "skin_disease", 6: "rotten_surface", 7: "wart",
}
LEATHER_CLASSES_CN = {
    0: "背景", 1: "开创伤", 2: "刺刮伤", 3: "烙印",
    4: "破洞", 5: "皮肤藓", 6: "烂面", 7: "刺猴",
}
NEU_CLASSES = {
    0: "background", 1: "crazing", 2: "inclusion", 3: "patches",
}
NEU_CLASSES_CN = {
    0: "背景", 1: "龟裂", 2: "夹杂物", 3: "补丁",
}


# ============================================================
# 颜色映射
# ============================================================
def get_colormap(n_classes):
    """生成类别颜色映射"""
    base_colors = [
        [0, 0, 0],        # 0: background - black
        [255, 0, 0],      # 1: red
        [0, 255, 0],      # 2: green
        [0, 0, 255],      # 3: blue
        [255, 255, 0],    # 4: yellow
        [255, 0, 255],    # 5: magenta
        [0, 255, 255],    # 6: cyan
        [128, 128, 0],    # 7: olive
    ]
    while len(base_colors) < n_classes:
        base_colors.append([np.random.randint(0, 255) for _ in range(3)])
    return np.array(base_colors[:n_classes], dtype=np.uint8)


def mask_to_rgb(mask, colormap):
    """将类别掩码转为 RGB 图像"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in enumerate(colormap):
        rgb[mask == c] = color
    return rgb


def overlay_mask(img_np, mask, colormap, alpha=0.5):
    """将掩码叠加到图像上"""
    mask_rgb = mask_to_rgb(mask, colormap)
    # 只在有缺陷的区域叠加
    defect_area = mask > 0
    blended = img_np.copy().astype(np.float32)
    for c in range(3):
        blended[:, :, c] = np.where(
            defect_area,
            alpha * mask_rgb[:, :, c] + (1 - alpha) * blended[:, :, c],
            blended[:, :, c]
        )
    return blended.astype(np.uint8)


# ============================================================
# 模型加载（复用）
# ============================================================
def load_module_from_file(filepath):
    spec = importlib.util.spec_from_file_location("model_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_b(weight_path, n_classes, device):
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50.py"))
    model = module.DSMONet(num_classes=n_classes, backbone=resnet50(),
                           backbone_indices=[0,1,2,3,4], out_ch=128).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_a2ms_defectnet_b(weight_path, n_classes, device):
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50_eSE_adapt_detailloss.py"))
    model = module.DSMONet(num_classes=n_classes, backbone=resnet50(),
                           backbone_indices=[0,1,2,3,4], out_ch=128).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_ppliteseg_b(weight_path, n_classes, device):
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50_ppliteseg.py"))
    model = module.DSMONet(num_classes=n_classes, backbone=resnet50(),
                           backbone_indices=[0,1,2,3,4], out_ch=128).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_stdc1_seg(weight_path, n_classes, device):
    module = load_module_from_file(os.path.join(NEW_COPY_DIR, "stdc.py"))
    model = module.BiSeNet(backbone='STDCNet1446', n_classes=n_classes,
                           pretrain_model='', use_boundary_8=True).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_ddrnet23slim(weight_path, n_classes, device):
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_ddr_s.py"))
    model = module.DSMONet(block=module.BasicBlock, layers=[2,2,2,2],
                           num_classes=n_classes, planes=32, spp_planes=128,
                           head_planes=64, augment=False).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


LOADER_MAP = {
    'load_base_b': load_base_b,
    'load_a2ms_defectnet_b': load_a2ms_defectnet_b,
    'load_ppliteseg_b': load_ppliteseg_b,
    'load_stdc1_seg': load_stdc1_seg,
    'load_ddrnet23slim': load_ddrnet23slim,
}


# ============================================================
# 数据集
# ============================================================
class LeatherDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path='dataset/pige/test.txt', input_hw=(768, 768)):
        self.input_hw = input_hw
        self.pairs = []
        self.img_paths = []
        self.ann_paths = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    ann = os.path.join('dataset', 'pige', parts[1])
                    img = os.path.join('dataset', 'pige', parts[0])
                    self.pairs.append((ann, img))
                    self.ann_paths.append(ann)
                    self.img_paths.append(img)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ann_path, img_path = self.pairs[idx]
        mask = Image.open(ann_path).convert('L')
        mask = mask.resize(self.input_hw, Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)

        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize(self.input_hw, Image.BILINEAR)
        img_np = np.array(img_resized)
        img_t = torch.from_numpy(img_np.astype(np.float32).transpose(2, 0, 1) / 255.0)

        mask_t = torch.from_numpy(mask_np)
        return img_t, mask_t, img_np, mask_np, idx


class NEUSegDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path='dataset/test_neu.txt', input_hw=(200, 200)):
        self.input_hw = input_hw
        self.pairs = []
        self.img_paths = []
        self.ann_paths = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    ann = os.path.join('dataset', parts[0])
                    img = os.path.join('dataset', parts[1])
                    self.pairs.append((ann, img))
                    self.ann_paths.append(ann)
                    self.img_paths.append(img)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ann_path, img_path = self.pairs[idx]
        mask = Image.open(ann_path).convert('L')
        mask = mask.resize(self.input_hw, Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)

        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize(self.input_hw, Image.BILINEAR)
        img_np = np.array(img_resized)
        img_t = torch.from_numpy(img_np.astype(np.float32).transpose(2, 0, 1) / 255.0)

        mask_t = torch.from_numpy(mask_np)
        return img_t, mask_t, img_np, mask_np, idx


# ============================================================
# 预测生成
# ============================================================
def predict_all_models(models, dataset, device, batch_size=4):
    """对所有样本生成各模型预测结果。返回 {model_name: [pred_mask, ...]}"""
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_preds = {name: [] for name in models}

    for name, model in models.items():
        model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", ncols=80):
            images = batch[0].to(device, dtype=torch.float)
            gt_masks = batch[3].numpy()  # mask_np
            indices = batch[4].numpy()

            for name, model in models.items():
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    pred = outputs[0]
                else:
                    pred = outputs
                pred_np = pred.data.max(1)[1].cpu().numpy()
                for p in pred_np:
                    all_preds[name].append(p)

    return all_preds


# ============================================================
# 样本筛选
# ============================================================
def compute_defect_stats(mask, n_classes):
    h, w = mask.shape
    total = h * w
    defect_mask = (mask > 0).astype(np.uint8)
    area_ratio = defect_mask.sum() / total
    labeled, n_comp = ndimage.label(defect_mask)
    max_ar = 0.0
    for rid in range(1, n_comp + 1):
        rm = (labeled == rid)
        if rm.sum() < 5:
            continue
        rows = np.any(rm, axis=1)
        cols = np.any(rm, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bh, bw = rmax - rmin + 1, cmax - cmin + 1
        ar = max(bh, bw) / (min(bh, bw) + 1e-6)
        max_ar = max(max_ar, ar)
    return area_ratio, n_comp, max_ar


def select_samples(dataset, all_preds, gt_masks, n_classes, classes_cn,
                   model_names, max_per_category=5):
    """筛选典型样本，返回 {category: [(idx, reason), ...]}"""
    categories = {
        'small_target': [],
        'elongated': [],
        'multi_defect': [],
        'texture_confusion': [],
        'base_miss_a2ms_hit': [],
        'base_boundary_break': [],
    }

    n = len(gt_masks)
    stats = []
    for i in range(n):
        ar, nc, mar = compute_defect_stats(gt_masks[i], n_classes)
        stats.append((ar, nc, mar))

    # 计算分位数
    defect_indices = [i for i in range(n) if stats[i][0] > 0]
    if not defect_indices:
        return categories

    defect_areas = [stats[i][0] for i in defect_indices]
    defect_ars = [stats[i][2] for i in defect_indices]
    p25 = np.percentile(defect_areas, 25)
    p75_ar = np.percentile(defect_ars, 75)

    for i in defect_indices:
        ar, nc, mar = stats[i]

        # 小目标
        if ar < p25 and len(categories['small_target']) < max_per_category:
            categories['small_target'].append((i, f"面积占比={ar:.4f}"))

        # 细长
        if mar > p75_ar and len(categories['elongated']) < max_per_category:
            categories['elongated'].append((i, f"长宽比={mar:.1f}"))

        # 多缺陷
        if nc > 1 and len(categories['multi_defect']) < max_per_category:
            categories['multi_defect'].append((i, f"连通域={nc}"))

    # 背景纹理干扰：无缺陷但模型预测出大量缺陷
    no_defect = [i for i in range(n) if stats[i][0] == 0]
    for i in no_defect:
        if len(categories['texture_confusion']) >= max_per_category:
            break
        for mname in model_names:
            if mname in all_preds:
                false_pos = (all_preds[mname][i] > 0).sum()
                total = all_preds[mname][i].size
                if false_pos / total > 0.05:
                    categories['texture_confusion'].append(
                        (i, f"无缺陷但{mname}误检率={false_pos/total:.2%}"))
                    break

    # Base-B 漏检但 A2MS 检出
    if 'Base-B' in all_preds and 'A2MS-DefectNet-B' in all_preds:
        for i in defect_indices:
            if len(categories['base_miss_a2ms_hit']) >= max_per_category:
                break
            gt_binary = (gt_masks[i] > 0)
            base_binary = (all_preds['Base-B'][i] > 0)
            a2ms_binary = (all_preds['A2MS-DefectNet-B'][i] > 0)
            # Base-B 漏检（Recall 低）但 A2MS 检出
            base_recall = (gt_binary & base_binary).sum() / (gt_binary.sum() + 1e-6)
            a2ms_recall = (gt_binary & a2ms_binary).sum() / (gt_binary.sum() + 1e-6)
            if base_recall < 0.5 and a2ms_recall > 0.7:
                categories['base_miss_a2ms_hit'].append(
                    (i, f"Base-B Recall={base_recall:.2f}, A2MS Recall={a2ms_recall:.2f}"))

        # Base-B 边界断裂
        for i in defect_indices:
            if len(categories['base_boundary_break']) >= max_per_category:
                break
            gt_binary = (gt_masks[i] > 0).astype(np.uint8)
            base_binary = (all_preds['Base-B'][i] > 0).astype(np.uint8)
            a2ms_binary = (all_preds['A2MS-DefectNet-B'][i] > 0).astype(np.uint8)
            # IoU 对比
            base_iou = (gt_binary & base_binary).sum() / ((gt_binary | base_binary).sum() + 1e-6)
            a2ms_iou = (gt_binary & a2ms_binary).sum() / ((gt_binary | a2ms_binary).sum() + 1e-6)
            # 边界完整性：连通域数量差异
            _, gt_nc = ndimage.label(gt_binary)
            _, base_nc = ndimage.label(base_binary)
            _, a2ms_nc = ndimage.label(a2ms_binary)
            if a2ms_iou - base_iou > 0.1 and abs(a2ms_nc - gt_nc) < abs(base_nc - gt_nc):
                categories['base_boundary_break'].append(
                    (i, f"Base IoU={base_iou:.2f}, A2MS IoU={a2ms_iou:.2f}"))

    return categories


# ============================================================
# 可视化
# ============================================================
def make_comparison_figure(img_np, gt_mask, pred_masks, model_names,
                           colormap, title, save_path):
    """生成对比图：原图 / GT / Model1 / Model2 / ..."""
    n_panels = 2 + len(model_names)  # img + gt + models
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # 原图
    axes[0].imshow(img_np)
    axes[0].set_title('Original', fontsize=9)
    axes[0].axis('off')

    # GT 叠加
    gt_overlay = overlay_mask(img_np, gt_mask, colormap, alpha=0.5)
    axes[1].imshow(gt_overlay)
    axes[1].set_title('GT', fontsize=9)
    axes[1].axis('off')

    # 各模型预测
    for j, (mname, pred) in enumerate(zip(model_names, pred_masks)):
        pred_overlay = overlay_mask(img_np, pred, colormap, alpha=0.5)
        axes[2 + j].imshow(pred_overlay)
        axes[2 + j].set_title(mname, fontsize=9)
        axes[2 + j].axis('off')

    fig.suptitle(title, fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='定性可视化')
    parser.add_argument('--dataset', type=str, default='leather',
                        choices=['leather', 'neu'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-per-category', type=int, default=5)
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    print(f"数据集: {args.dataset}")

    # 配置
    if args.dataset == 'leather':
        n_classes = 8
        classes = LEATHER_CLASSES
        classes_cn = LEATHER_CLASSES_CN
        dataset = LeatherDataset()
        weight_map = {
            'Base-B': ('load_base_b', os.path.join(NEW_COPY_DIR, "model_savePath", "dsmonet_resnet_pascal_pige_dsmor50_0126.pkl")),
            'A2MS-DefectNet-B': ('load_a2ms_defectnet_b', os.path.join(NEW_DIR, "dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl")),
            'PP-LiteSeg-B': ('load_ppliteseg_b', None),  # 缺少权重
            'STDC1-Seg': ('load_stdc1_seg', os.path.join(NEW_COPY_DIR, "model_savePath", "sdtdcnet_pige_stdc2_pige.pkl")),
            'DDRNet23slim': ('load_ddrnet23slim', None),  # 缺少权重
        }
    else:
        n_classes = 4
        classes = NEU_CLASSES
        classes_cn = NEU_CLASSES_CN
        dataset = NEUSegDataset()
        weight_map = {}  # 待上传权重

    colormap = get_colormap(n_classes)

    # 加载可用模型
    models = {}
    model_order = []
    for mname, (loader_name, wpath) in weight_map.items():
        if wpath and os.path.exists(wpath):
            print(f"加载 {mname}...")
            try:
                models[mname] = LOADER_MAP[loader_name](wpath, n_classes, device)
                model_order.append(mname)
            except Exception as e:
                print(f"  失败: {e}")

    if not models:
        print("没有可用模型，退出")
        return

    # 生成所有预测
    print("生成预测...")
    all_preds = predict_all_models(models, dataset, device)

    # 加载 GT
    gt_masks = []
    for i in range(len(dataset)):
        _, _, _, mask_np, _ = dataset[i]
        gt_masks.append(mask_np)

    # 筛选样本
    print("筛选典型样本...")
    categories = select_samples(dataset, all_preds, gt_masks, n_classes, classes_cn,
                                model_order, max_per_category=args.max_per_category)

    # 生成可视化
    out_dir = os.path.join(WORKSPACE, "Industrial-Surface-Defect", "figures", "qualitative_results")
    os.makedirs(out_dir, exist_ok=True)

    category_names = {
        'small_target': '小目标缺陷',
        'elongated': '细长划痕',
        'multi_defect': '多缺陷样本',
        'texture_confusion': '背景纹理干扰',
        'base_miss_a2ms_hit': 'Base-B漏检_A2MS检出',
        'base_boundary_break': 'Base-B边界断裂_A2MS完整',
    }

    readme_lines = ["# 定性可视化结果\n"]
    readme_lines.append(f"- 日期: {time.strftime('%Y-%m-%d')}\n")
    readme_lines.append(f"- 数据集: {args.dataset}\n")
    readme_lines.append(f"- 模型: {', '.join(model_order)}\n\n")

    total_figures = 0
    for cat_key, samples in categories.items():
        cat_name = category_names.get(cat_key, cat_key)
        if not samples:
            readme_lines.append(f"## {cat_name}\n\n无符合条件的样本。\n\n")
            continue

        readme_lines.append(f"## {cat_name}\n\n")
        readme_lines.append("| 样本ID | 文件名 | 原因 |\n")
        readme_lines.append("|---|---|---|\n")

        for idx, reason in samples:
            img_np = dataset[idx][2]
            gt_mask = gt_masks[idx]
            preds = [all_preds[m][idx] for m in model_order]

            # 计算该样本的 mIoU
            iou_info = []
            for m in model_order:
                pred_binary = (all_preds[m][idx] > 0).astype(np.uint8)
                gt_binary = (gt_mask > 0).astype(np.uint8)
                iou = (gt_binary & pred_binary).sum() / ((gt_binary | pred_binary).sum() + 1e-6)
                iou_info.append(f"{m} IoU={iou:.2f}")

            fname = f"{cat_key}_idx{idx:04d}.png"
            save_path = os.path.join(out_dir, fname)

            # 获取原始图像路径
            orig_path = dataset.img_paths[idx] if hasattr(dataset, 'img_paths') else f"idx_{idx}"

            make_comparison_figure(
                img_np, gt_mask, preds, model_order, colormap,
                title=f"{cat_name} | ID={idx} | {'; '.join(iou_info)}",
                save_path=save_path
            )

            readme_lines.append(f"| {idx} | {fname} | {reason} |\n")
            total_figures += 1

        readme_lines.append("\n")

    # 保存 README
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.writelines(readme_lines)

    # 保存样本索引映射
    index_path = os.path.join(out_dir, "sample_index.json")
    index_data = {}
    for cat_key, samples in categories.items():
        index_data[cat_key] = [
            {
                'dataset_index': int(idx),
                'image_path': dataset.img_paths[idx] if hasattr(dataset, 'img_paths') else '',
                'annotation_path': dataset.ann_paths[idx] if hasattr(dataset, 'ann_paths') else '',
                'reason': reason,
            }
            for idx, reason in samples
        ]
    with open(index_path, 'w') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    # 释放显存
    for m in models.values():
        del m
    torch.cuda.empty_cache()

    print(f"\n生成 {total_figures} 张对比图")
    print(f"输出目录: {out_dir}")
    print(f"README: {readme_path}")
    print(f"样本索引: {index_path}")
    print("完成！")


if __name__ == '__main__':
    import time
    main()

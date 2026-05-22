#!/usr/bin/env python3
"""
分组评价：按缺陷面积占比、长宽比、连通域数量分组，评估各模型在不同子集上的表现。

分组规则：
- 小目标：缺陷面积占图像面积 < P25（仅含缺陷样本）
- 中等目标：P25 ~ P75
- 大目标：> P75
- 细长缺陷：外接框长宽比 > P75
- 多缺陷样本：连通域数 > 1
- 无缺陷样本：缺陷面积 = 0

输出：
- experiments/results/small_object_group_results.csv
- experiments/results/small_object_group_summary.md
"""

import sys
import os
import csv
import time
import json
import argparse
import numpy as np
import torch
import importlib.util
from scipy import ndimage
from torch.utils import data
from tqdm import tqdm
from PIL import Image

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
# 连通域分析
# ============================================================
def compute_defect_statistics(mask, n_classes):
    """计算单张图像的缺陷统计特征。

    注：标签为语义掩码（非实例级），连通域数量为近似统计。
    """
    h, w = mask.shape
    total_area = h * w
    defect_mask = (mask > 0).astype(np.uint8)
    defect_area = defect_mask.sum()
    area_ratio = defect_area / total_area

    # 连通域
    labeled, n_comp = ndimage.label(defect_mask)

    # 外接框长宽比（对每个连通域）
    aspect_ratios = []
    n_instances = 0
    for region_id in range(1, n_comp + 1):
        region_mask = (labeled == region_id)
        region_area = region_mask.sum()
        if region_area < 5:
            continue
        n_instances += 1
        rows = np.any(region_mask, axis=1)
        cols = np.any(region_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bh = rmax - rmin + 1
        bw = cmax - cmin + 1
        aspect_ratios.append(max(bh, bw) / (min(bh, bw) + 1e-6))

    max_aspect = max(aspect_ratios) if aspect_ratios else 0.0

    return {
        'area_ratio': area_ratio,
        'n_components': n_comp,
        'n_instances': n_instances,
        'max_aspect_ratio': max_aspect,
    }


def load_annotations(txt_path, base_dir):
    """加载所有标注掩码，返回 (ann_path, img_path, mask) 列表。"""
    pairs = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))

    results = []
    for ann_rel, img_rel in pairs:
        ann_path = os.path.join(base_dir, ann_rel)
        img_path = os.path.join(base_dir, img_rel)
        mask = np.array(Image.open(ann_path).convert('L'))
        results.append((ann_path, img_path, mask))
    return results


def compute_grouping(stats_list):
    """基于统计特征计算分组阈值并分配组别。

    仅对有缺陷的样本计算面积分位数。
    """
    area_ratios = np.array([s['area_ratio'] for s in stats_list])
    has_defect = area_ratios > 0

    if has_defect.sum() > 0:
        defect_areas = area_ratios[has_defect]
        p25 = np.percentile(defect_areas, 25)
        p75 = np.percentile(defect_areas, 75)
    else:
        p25, p75 = 0.0, 0.0

    aspect_ratios = np.array([s['max_aspect_ratio'] for s in stats_list])
    defect_aspects = aspect_ratios[has_defect] if has_defect.sum() > 0 else np.array([0])
    ar_p75 = np.percentile(defect_aspects, 75)

    thresholds = {
        'area_p25': p25,
        'area_p75': p75,
        'aspect_p75': ar_p75,
    }

    groups = []
    for i, s in enumerate(stats_list):
        grp = set()
        if s['area_ratio'] == 0:
            grp.add('no_defect')
        else:
            if s['area_ratio'] < p25:
                grp.add('small')
            elif s['area_ratio'] < p75:
                grp.add('medium')
            else:
                grp.add('large')

        if s['max_aspect_ratio'] > ar_p75 and s['area_ratio'] > 0:
            grp.add('elongated')

        if s['n_instances'] > 1:
            grp.add('multi_defect')

        groups.append(grp)

    return groups, thresholds


# ============================================================
# 指标计算
# ============================================================
def compute_metrics_from_confusion(confusion_matrix):
    hist = confusion_matrix
    n_classes = hist.shape[0]
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
    dice = 2 * iu / (1 + iu + 1e-10)
    recall = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
    precision = np.diag(hist) / (hist.sum(axis=0) + 1e-10)
    miou = np.nanmean(iu)
    return {'iou': iu, 'dice': dice, 'recall': recall, 'precision': precision, 'miou': miou}


class runningScore:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        mean_iu = np.nanmean(iu)
        cls_iu = dict(zip(range(self.n_classes), iu))
        return {"Overall Acc": acc, "Mean IoU": mean_iu}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


# ============================================================
# 模型加载（复用 evaluate_class_iou.py 的逻辑）
# ============================================================
def load_module_from_file(filepath):
    spec = importlib.util.spec_from_file_location("model_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_b(weight_path, n_classes, device='cuda'):
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50.py"))
    model = module.DSMONet(num_classes=n_classes, backbone=resnet50(),
                           backbone_indices=[0,1,2,3,4], out_ch=128).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_a2ms_defectnet_b(weight_path, n_classes, device='cuda'):
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50_eSE_adapt_detailloss.py"))
    model = module.DSMONet(num_classes=n_classes, backbone=resnet50(),
                           backbone_indices=[0,1,2,3,4], out_ch=128).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_base_s(weight_path, n_classes, device='cuda'):
    from model_resnet import resnet18
    module = load_module_from_file(os.path.join(NEW_COPY_DIR, "model_dsmo_rs18.py"))
    model = module.DSMONet(num_classes=n_classes, backbone=resnet18(),
                           backbone_indices=[1,2,3,4], out_ch=128).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_a2ms_defectnet_s(weight_path, n_classes, device='cuda'):
    from model_resnet import resnet18
    module = load_module_from_file(os.path.join(NEW_COPY_DIR, "model_dsmo_rs18_eSE_adapt_detailloss_822.py"))
    model = module.DSMONet(num_classes=n_classes, backbone=resnet18(),
                           backbone_indices=[1,2,3,4], out_ch=128).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_ppliteseg_b(weight_path, n_classes, device='cuda'):
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50_ppliteseg.py"))
    model = module.DSMONet(num_classes=n_classes, backbone=resnet50(),
                           backbone_indices=[0,1,2,3,4], out_ch=128).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_stdc1_seg(weight_path, n_classes, device='cuda'):
    module = load_module_from_file(os.path.join(NEW_COPY_DIR, "stdc.py"))
    model = module.BiSeNet(backbone='STDCNet1446', n_classes=n_classes,
                           pretrain_model='', use_boundary_8=True).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_ddrnet23slim(weight_path, n_classes, device='cuda'):
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_ddr_s.py"))
    model = module.DSMONet(block=module.BasicBlock, layers=[2,2,2,2],
                           num_classes=n_classes, planes=32, spp_planes=128,
                           head_planes=64, augment=False).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_pidnet_s(weight_path, n_classes, device='cuda'):
    module = load_module_from_file(os.path.join(NEW_DIR, "pid.py"))
    model = module.PIDNet(m=2, n=3, num_classes=n_classes, planes=32,
                          ppm_planes=96, head_planes=128, augment=True).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


def load_subregion_unet(weight_path, n_classes, device='cuda'):
    module = load_module_from_file(os.path.join(WORKSPACE, "subregion unet", "model_p.py"))
    model = module.fianlModel(inchannel=3, nclass=n_classes).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model


# ============================================================
# 数据集定义
# ============================================================
class LeatherDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path='dataset/pige/test.txt', input_hw=(768, 768)):
        self.input_hw = input_hw
        self.pairs = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.pairs.append((
                        os.path.join('dataset', 'pige', parts[1]),  # label
                        os.path.join('dataset', 'pige', parts[0]),  # image
                    ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ann_path, img_path = self.pairs[idx]
        mask = Image.open(ann_path).convert('L')
        mask = mask.resize(self.input_hw, Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)

        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.input_hw, Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))

        mask_t = torch.from_numpy(mask)
        binary = (mask_t > 0).long()
        return img, mask_t, binary


class NEUSegDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path='dataset/test_neu.txt', input_hw=(200, 200)):
        self.input_hw = input_hw
        self.pairs = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.pairs.append((
                        os.path.join('dataset', parts[0]),  # annotation
                        os.path.join('dataset', parts[1]),  # image
                    ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ann_path, img_path = self.pairs[idx]
        mask = Image.open(ann_path).convert('L')
        mask = mask.resize(self.input_hw, Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)

        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.input_hw, Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))

        mask_t = torch.from_numpy(mask)
        binary = (mask_t > 0).long()
        return img, mask_t, binary


# ============================================================
# 模型配置
# ============================================================
LEATHER_WEIGHTS = {
    'Base-B': ('load_base_b', os.path.join(NEW_COPY_DIR, "model_savePath", "dsmonet_resnet_pascal_pige_dsmor50_0126.pkl")),
    'A2MS-DefectNet-B': ('load_a2ms_defectnet_b', os.path.join(NEW_DIR, "dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl")),
    'STDC1-Seg': ('load_stdc1_seg', os.path.join(NEW_COPY_DIR, "model_savePath", "sdtdcnet_pige_stdc2_pige.pkl")),
}

NEU_WEIGHTS = {
    # 待上传权重后填入
    # 'Base-B': ('load_base_b', '/path/to/neu_base_b.pkl'),
    # 'A2MS-DefectNet-B': ('load_a2ms_defectnet_b', '/path/to/neu_a2ms_b.pkl'),
}

LOADER_MAP = {
    'load_base_b': load_base_b,
    'load_a2ms_defectnet_b': load_a2ms_defectnet_b,
    'load_base_s': load_base_s,
    'load_a2ms_defectnet_s': load_a2ms_defectnet_s,
    'load_ppliteseg_b': load_ppliteseg_b,
    'load_stdc1_seg': load_stdc1_seg,
    'load_ddrnet23slim': load_ddrnet23slim,
    'load_pidnet_s': load_pidnet_s,
    'load_subregion_unet': load_subregion_unet,
}


# ============================================================
# 评估
# ============================================================
def evaluate_indices(model, dataset, indices, n_classes, device='cuda', batch_size=4):
    """对指定索引的样本评估，返回混淆矩阵和每样本预测。"""
    from torch.utils.data import Subset, DataLoader
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    metrics = runningScore(n_classes)
    sample_preds = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Eval", ncols=80, leave=False):
            images = images.to(device, dtype=torch.float)
            labels = labels.squeeze(1).to(device, dtype=torch.int64)

            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                pred = outputs[0]
            else:
                pred = outputs

            pred_np = pred.data.max(1)[1].cpu().numpy()
            gt_np = labels.data.cpu().numpy()
            metrics.update(gt_np, pred_np)

            for p in pred_np:
                sample_preds.append(p)

    score, cls_iu = metrics.get_scores()
    result = compute_metrics_from_confusion(metrics.confusion_matrix)
    return score, result, sample_preds


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='分组评价')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['leather', 'neu', 'all'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    print(f"Device: {device}, GPU: {gpu_name}")
    print(f"日期: {time.strftime('%Y-%m-%d')}")

    datasets_to_run = []
    if args.dataset in ('leather', 'all'):
        datasets_to_run.append('leather')
    if args.dataset in ('neu', 'all'):
        datasets_to_run.append('neu')

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'experiments', 'results'
    )
    os.makedirs(results_dir, exist_ok=True)

    all_csv_rows = []
    all_summaries = []

    for ds_key in datasets_to_run:
        if ds_key == 'leather':
            ds_name = 'Leather'
            n_classes = 8
            classes = LEATHER_CLASSES
            classes_cn = LEATHER_CLASSES_CN
            dataset = LeatherDataset()
            txt_path = 'dataset/pige/test.txt'
            base_dir = 'dataset/pige/'
            weight_map = LEATHER_WEIGHTS
            # Leather: test.txt 列顺序是 image label
            col_order = ('image', 'label')
        else:
            ds_name = 'NEU-Seg'
            n_classes = 4
            classes = NEU_CLASSES
            classes_cn = NEU_CLASSES_CN
            dataset = NEUSegDataset()
            txt_path = 'dataset/test_neu.txt'
            base_dir = 'dataset/'
            weight_map = NEU_WEIGHTS
            col_order = ('annotation', 'image')

        print(f"\n{'#'*60}")
        print(f"# 数据集: {ds_name}")
        print(f"{'#'*60}")

        # 加载标注并计算统计
        print("计算缺陷统计特征...")
        raw_pairs = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    if col_order[0] == 'image':
                        raw_pairs.append((parts[1], parts[0]))  # (ann, img)
                    else:
                        raw_pairs.append((parts[0], parts[1]))

        stats_list = []
        for ann_rel, img_rel in raw_pairs:
            ann_path = os.path.join(base_dir, ann_rel)
            mask = np.array(Image.open(ann_path).convert('L'))
            stats_list.append(compute_defect_statistics(mask, n_classes))

        groups, thresholds = compute_grouping(stats_list)

        print(f"分组阈值: area P25={thresholds['area_p25']:.4f}, P75={thresholds['area_p75']:.4f}, aspect P75={thresholds['aspect_p75']:.2f}")

        # 统计各组样本数
        group_names = ['small', 'medium', 'large', 'elongated', 'multi_defect', 'no_defect']
        group_labels = {
            'small': '小目标', 'medium': '中等目标', 'large': '大目标',
            'elongated': '细长缺陷', 'multi_defect': '多缺陷', 'no_defect': '无缺陷'
        }
        group_indices = {}
        for gn in group_names:
            idxs = [i for i, g in enumerate(groups) if gn in g]
            group_indices[gn] = idxs
            print(f"  {group_labels[gn]} ({gn}): {len(idxs)} 张")

        # 逐模型评估
        for model_name, (loader_name, weight_path) in weight_map.items():
            if not os.path.exists(weight_path):
                print(f"\n  [{model_name}] 权重不存在，跳过")
                for gn in group_names:
                    all_csv_rows.append({
                        'dataset': ds_name, 'model': model_name,
                        'group': gn, 'group_cn': group_labels[gn],
                        'n_samples': len(group_indices[gn]),
                        'miou': 'N/A', 'dice': 'N/A', 'recall': 'N/A',
                        'precision': 'N/A',
                        'miou_vs_base_b': 'N/A',
                    })
                continue

            print(f"\n  [{model_name}] 加载模型...")
            try:
                loader_fn = LOADER_MAP[loader_name]
                model = loader_fn(weight_path, n_classes, device)
            except Exception as e:
                print(f"  [错误] 加载失败: {e}")
                for gn in group_names:
                    all_csv_rows.append({
                        'dataset': ds_name, 'model': model_name,
                        'group': gn, 'group_cn': group_labels[gn],
                        'n_samples': len(group_indices[gn]),
                        'miou': 'N/A', 'dice': 'N/A', 'recall': 'N/A',
                        'precision': 'N/A', 'miou_vs_base_b': 'N/A',
                    })
                continue

            group_results = {}
            for gn in group_names:
                idxs = group_indices[gn]
                if len(idxs) == 0:
                    group_results[gn] = None
                    continue
                print(f"    {group_labels[gn]} ({len(idxs)} 张)...", end=' ')
                score, metrics, _ = evaluate_indices(model, dataset, idxs, n_classes, device)
                group_results[gn] = metrics
                print(f"mIoU={metrics['miou']:.4f}")

            # 记录结果
            base_b_results = {}
            if model_name == 'Base-B':
                base_b_results = {gn: r for gn, r in group_results.items() if r is not None}

            for gn in group_names:
                gr = group_results.get(gn)
                if gr is None:
                    all_csv_rows.append({
                        'dataset': ds_name, 'model': model_name,
                        'group': gn, 'group_cn': group_labels[gn],
                        'n_samples': len(group_indices[gn]),
                        'miou': 'N/A', 'dice': 'N/A', 'recall': 'N/A',
                        'precision': 'N/A', 'miou_vs_base_b': 'N/A',
                    })
                else:
                    delta = ''
                    if gn in base_b_results and model_name != 'Base-B':
                        delta = f"{gr['miou'] - base_b_results[gn]['miou']:+.4f}"
                    elif model_name == 'Base-B':
                        delta = '0.0000'

                    all_csv_rows.append({
                        'dataset': ds_name, 'model': model_name,
                        'group': gn, 'group_cn': group_labels[gn],
                        'n_samples': len(group_indices[gn]),
                        'miou': f"{gr['miou']:.4f}",
                        'dice': f"{np.nanmean(gr['dice']):.4f}",
                        'recall': f"{np.nanmean(gr['recall']):.4f}",
                        'precision': f"{np.nanmean(gr['precision']):.4f}",
                        'miou_vs_base_b': delta,
                    })

            del model
            torch.cuda.empty_cache()

        all_summaries.append({
            'dataset': ds_name,
            'n_classes': n_classes,
            'classes_cn': classes_cn,
            'thresholds': thresholds,
            'group_indices': {gn: len(idxs) for gn, idxs in group_indices.items()},
        })

    # 保存 CSV
    csv_path = os.path.join(results_dir, 'small_object_group_results.csv')
    fieldnames = ['dataset', 'model', 'group', 'group_cn', 'n_samples',
                  'miou', 'dice', 'recall', 'precision', 'miou_vs_base_b']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_csv_rows:
            writer.writerow(r)
    print(f"\nCSV 已保存: {csv_path}")

    # 保存 Markdown
    md_path = os.path.join(results_dir, 'small_object_group_summary.md')
    with open(md_path, 'w') as f:
        f.write("# 小目标与细长缺陷分组评价汇总\n\n")
        f.write(f"- 日期: {time.strftime('%Y-%m-%d')}\n")
        f.write(f"- 设备: {gpu_name}\n")
        f.write(f"- 脚本: `tools/evaluate_small_object_groups.py`\n\n")

        f.write("## 分组规则\n\n")
        f.write("| 组别 | 定义 |\n")
        f.write("|---|---|\n")
        f.write("| 小目标 | 缺陷面积占图像面积 < P25（仅含缺陷样本） |\n")
        f.write("| 中等目标 | P25 ≤ 面积占比 < P75 |\n")
        f.write("| 大目标 | 面积占比 ≥ P75 |\n")
        f.write("| 细长缺陷 | 外接框长宽比 > P75 且有缺陷 |\n")
        f.write("| 多缺陷 | 连通域数 > 1 |\n")
        f.write("| 无缺陷 | 缺陷面积 = 0 |\n\n")
        f.write("注：标签为语义掩码（非实例级），连通域数量基于二值缺陷掩码的连通域分析近似统计。\n\n")

        for summary in all_summaries:
            ds_name = summary['dataset']
            th = summary['thresholds']
            f.write(f"## {ds_name}\n\n")
            f.write(f"- 面积占比 P25: {th['area_p25']:.4f}\n")
            f.write(f"- 面积占比 P75: {th['area_p75']:.4f}\n")
            f.write(f"- 长宽比 P75: {th['aspect_p75']:.2f}\n\n")

            f.write("### 样本数分布\n\n")
            f.write("| 组别 | 样本数 |\n")
            f.write("|---|---|\n")
            for gn, cnt in summary['group_indices'].items():
                f.write(f"| {group_labels[gn]} | {cnt} |\n")

            # 模型对比表
            ds_rows = [r for r in all_csv_rows if r['dataset'] == ds_name]
            model_names = list(dict.fromkeys(r['model'] for r in ds_rows))

            f.write("\n### 各组 mIoU 对比\n\n")
            header = "| 组别 | 样本数 |"
            sep = "|---|---|"
            for m in model_names:
                header += f" {m} |"
                sep += "---|"
            header += " Base-B 差值(最佳) |"
            sep += "---|"
            f.write(header + "\n" + sep + "\n")

            for gn in group_names:
                gn_rows = [r for r in ds_rows if r['group'] == gn]
                if not gn_rows:
                    continue
                n_samp = gn_rows[0]['n_samples']
                row = f"| {group_labels[gn]} | {n_samp} |"
                best_delta = ''
                best_val = -1
                for m in model_names:
                    mr = [r for r in gn_rows if r['model'] == m]
                    if mr:
                        val = mr[0]['miou']
                        row += f" {val} |"
                        if val != 'N/A' and m != 'Base-B':
                            try:
                                v = float(val)
                                if v > best_val:
                                    best_val = v
                                    best_delta = mr[0]['miou_vs_base_b']
                            except ValueError:
                                pass
                    else:
                        row += " N/A |"
                row += f" {best_delta} |"
                f.write(row + "\n")

            f.write("\n### 分析要点\n\n")
            f.write("- 小目标缺陷：面积占比最低的 25% 缺陷样本，检验模型对微小缺陷的检出能力。\n")
            f.write("- 细长缺陷：长宽比最高的 25% 缺陷样本，检验模型对条状/线状缺陷的分割连续性。\n")
            f.write("- 多缺陷：同一图像包含多个缺陷连通域，检验模型对多实例的区分能力。\n")

    print(f"汇总已保存: {md_path}")
    print("\n完成！")


if __name__ == '__main__':
    main()

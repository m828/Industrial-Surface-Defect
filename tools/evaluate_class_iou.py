#!/usr/bin/env python3
"""
类别级 IoU 评估脚本：在皮革缺陷数据集上评估各模型的 per-class IoU/Dice/Recall/Precision。

使用方法：
    cd /workspace/Industrial\\ Surface\\ Defect/new
    /opt/conda/bin/python /workspace/Industrial\\ Surface\\ Defect/Industrial-Surface-Defect/tools/evaluate_class_iou.py

输出：
    - 终端表格
    - CSV: experiments/results/class_iou_results.csv
    - 汇总: experiments/results/class_iou_summary.md

约束：
    - 不修改模型结构
    - 不编造数据
    - 缺少权重的模型标注"缺少权重，未评估"
    - 类别名称来自数据加载脚本
"""

import sys
import os
import csv
import time
import argparse
import numpy as np
import torch
import importlib.util
from torch.utils import data
from tqdm import tqdm

# ============================================================
# 路径配置
# ============================================================
WORKSPACE = "/workspace/Industrial Surface Defect"
NEW_DIR = os.path.join(WORKSPACE, "new")
NEW_COPY_DIR = os.path.join(WORKSPACE, "new (copy)")
SUBREGION_DIR = os.path.join(WORKSPACE, "subregion unet")

sys.path.insert(0, NEW_DIR)
sys.path.insert(0, NEW_COPY_DIR)

# ============================================================
# 类别定义（来自 datagenerator_yachi.py 和论文）
# ============================================================
LEATHER_CLASSES = {
    0: "background",
    1: "open_wound",      # 开创伤
    2: "scratch",          # 刺刮伤
    3: "brand_mark",       # 烙印
    4: "hole",             # 破洞
    5: "skin_disease",     # 皮肤藓
    6: "rotten_surface",   # 烂面
    7: "wart",             # 刺猴
}

LEATHER_CLASSES_CN = {
    0: "背景",
    1: "开创伤",
    2: "刺刮伤",
    3: "烙印",
    4: "破洞",
    5: "皮肤藓",
    6: "烂面",
    7: "刺猴",
}

NEU_CLASSES = {
    0: "background",
    1: "crazing",      # 龟裂
    2: "inclusion",    # 夹杂物
    3: "patches",      # 补丁
}

NEU_CLASSES_CN = {
    0: "背景",
    1: "龟裂",
    2: "夹杂物",
    3: "补丁",
}


# ============================================================
# 指标计算
# ============================================================
def compute_metrics_from_confusion(confusion_matrix):
    """从混淆矩阵计算 per-class IoU, Dice, Recall, Precision"""
    n_classes = confusion_matrix.shape[0]
    hist = confusion_matrix

    # Per-class IoU: TP / (TP + FP + FN)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)

    # Per-class Dice: 2*TP / (2*TP + FP + FN) = 2*IoU / (1+IoU)
    dice = 2 * iu / (1 + iu + 1e-10)

    # Per-class Recall: TP / (TP + FN) = diagonal / row_sum
    recall = np.diag(hist) / (hist.sum(axis=1) + 1e-10)

    # Per-class Precision: TP / (TP + FP) = diagonal / col_sum
    precision = np.diag(hist) / (hist.sum(axis=0) + 1e-10)

    # mIoU
    miou = np.nanmean(iu)

    return {
        'iou': iu,
        'dice': dice,
        'recall': recall,
        'precision': precision,
        'miou': miou,
    }


# ============================================================
# runningScore（从 tools/computemIou.py 复制，避免导入问题）
# ============================================================
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
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / (hist.sum() + 1e-10)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
        }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


# ============================================================
# 动态加载模块
# ============================================================
def load_module_from_file(filepath):
    spec = importlib.util.spec_from_file_location("model_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# 模型工厂
# ============================================================
def load_a2ms_defectnet_b(weight_path, n_classes=8, device='cuda'):
    """A2MS-DefectNet-B: ResNet-50 + eSE + AdaptiveChannelWeight + detailloss"""
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50_eSE_adapt_detailloss.py"))
    model = module.DSMONet(
        num_classes=n_classes,
        backbone=resnet50(),
        backbone_indices=[0, 1, 2, 3, 4],
        out_ch=128
    ).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_base_b(weight_path, n_classes=8, device='cuda'):
    """Base-B: ResNet-50 + SELayer"""
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50.py"))
    model = module.DSMONet(
        num_classes=n_classes,
        backbone=resnet50(),
        backbone_indices=[0, 1, 2, 3, 4],
        out_ch=128
    ).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_stdc1_seg(weight_path, n_classes=8, device='cuda'):
    """STDC1-Seg: BiSeNet + STDCNet1446（权重文件虽名为stdc2，实际为STDCNet1446）"""
    module = load_module_from_file(os.path.join(NEW_COPY_DIR, "stdc.py"))
    model = module.BiSeNet(
        backbone='STDCNet1446',
        n_classes=n_classes,
        pretrain_model='',
        use_boundary_8=True
    ).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_base_s(weight_path, n_classes=8, device='cuda'):
    """Base-S: ResNet-18 + SELayer"""
    from model_resnet import resnet18
    module = load_module_from_file(os.path.join(NEW_COPY_DIR, "model_dsmo_rs18.py"))
    model = module.DSMONet(
        num_classes=n_classes,
        backbone=resnet18(),
        backbone_indices=[1, 2, 3, 4],
        out_ch=128
    ).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_a2ms_defectnet_s(weight_path, n_classes=8, device='cuda'):
    """A2MS-DefectNet-S: ResNet-18 + eSE + AdaptiveChannelWeight + detailloss"""
    from model_resnet import resnet18
    module = load_module_from_file(os.path.join(NEW_COPY_DIR, "model_dsmo_rs18_eSE_adapt_detailloss_822.py"))
    model = module.DSMONet(
        num_classes=n_classes,
        backbone=resnet18(),
        backbone_indices=[1, 2, 3, 4],
        out_ch=128
    ).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_ppliteseg_b(weight_path, n_classes=8, device='cuda'):
    """PP-LiteSeg-B: ResNet-50 + SPPM"""
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50_ppliteseg.py"))
    model = module.DSMONet(
        num_classes=n_classes,
        backbone=resnet50(),
        backbone_indices=[0, 1, 2, 3, 4],
        out_ch=128
    ).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_ddrnet23slim(weight_path, n_classes=8, device='cuda'):
    """DDRNet23slim"""
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_ddr_s.py"))
    model = module.DSMONet(
        block=module.BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=n_classes,
        planes=32,
        spp_planes=128,
        head_planes=64,
        augment=False
    ).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_bisenetv2(weight_path, n_classes=8, device='cuda'):
    """BiSeNetV2-L: 需从外部引入"""
    raise NotImplementedError("BiSeNetV2 代码文件未在服务器上找到")


def load_subregion_unet(weight_path, n_classes=8, device='cuda'):
    """Sub-region UNet: fianlModel"""
    module = load_module_from_file(os.path.join(SUBREGION_DIR, "model_p.py"))
    model = module.fianlModel(inchannel=3, nclass=n_classes).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def load_pidnet_s(weight_path, n_classes=8, device='cuda'):
    """PIDNet-S"""
    module = load_module_from_file(os.path.join(NEW_DIR, "pid.py"))
    model = module.PIDNet(
        m=2, n=3, num_classes=n_classes,
        planes=32, ppm_planes=96, head_planes=128, augment=True
    ).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


# ============================================================
# 模型配置表
# ============================================================
MODEL_CONFIGS = [
    {
        'name': 'Base-B',
        'dataset': 'Leather',
        'n_classes': 8,
        'loader': load_base_b,
        'weight_path': os.path.join(NEW_COPY_DIR, "model_savePath", "dsmonet_resnet_pascal_pige_dsmor50_0126.pkl"),
        'backbone': 'ResNet-50',
    },
    {
        'name': 'A2MS-DefectNet-B',
        'dataset': 'Leather',
        'n_classes': 8,
        'loader': load_a2ms_defectnet_b,
        'weight_path': os.path.join(NEW_DIR, "dsmonet_resnet_pascal_pige_dsmor50_eSE_adapt_detailloss_160000.pkl"),
        'backbone': 'ResNet-50',
    },
    {
        'name': 'Base-S',
        'dataset': 'Leather',
        'n_classes': 8,
        'loader': load_base_s,
        'weight_path': None,  # 缺少权重
        'backbone': 'ResNet-18',
    },
    {
        'name': 'A2MS-DefectNet-S',
        'dataset': 'Leather',
        'n_classes': 8,
        'loader': load_a2ms_defectnet_s,
        'weight_path': None,  # 缺少权重
        'backbone': 'ResNet-18',
    },
    {
        'name': 'PP-LiteSeg-B',
        'dataset': 'Leather',
        'n_classes': 8,
        'loader': load_ppliteseg_b,
        'weight_path': None,  # 缺少权重
        'backbone': 'ResNet-50',
    },
    {
        'name': 'STDC1-Seg',
        'dataset': 'Leather',
        'n_classes': 8,
        'loader': load_stdc1_seg,
        'weight_path': os.path.join(NEW_COPY_DIR, "model_savePath", "sdtdcnet_pige_stdc2_pige.pkl"),
        'backbone': 'STDCNet1446',
    },
    {
        'name': 'DDRNet23slim',
        'dataset': 'Leather',
        'n_classes': 8,
        'loader': load_ddrnet23slim,
        'weight_path': None,  # 缺少权重
        'backbone': 'DDRNet-Internal',
    },
    {
        'name': 'BiSeNetV2-L',
        'dataset': 'Leather',
        'n_classes': 8,
        'loader': load_bisenetv2,
        'weight_path': None,  # 缺少权重和代码
        'backbone': 'BiSeNetV2',
    },
    {
        'name': 'Sub-region UNet',
        'dataset': 'Leather',
        'n_classes': 8,
        'loader': load_subregion_unet,
        'weight_path': None,  # 仅有 NEU 权重，无 Leather 权重
        'backbone': 'Sub-region-UNet',
    },
    {
        'name': 'PIDNet-S',
        'dataset': 'Leather',
        'n_classes': 8,
        'loader': load_pidnet_s,
        'weight_path': None,  # 缺少权重
        'backbone': 'PIDNet-Internal',
    },
]

# NEU-Seg 模型配置（权重路径待确认后填入）
NEU_MODEL_CONFIGS = [
    {
        'name': 'Base-B',
        'dataset': 'NEU-Seg',
        'n_classes': 4,
        'loader': load_base_b,
        'weight_path': None,  # 缺少 NEU 权重，待上传
        'backbone': 'ResNet-50',
    },
    {
        'name': 'A2MS-DefectNet-B',
        'dataset': 'NEU-Seg',
        'n_classes': 4,
        'loader': load_a2ms_defectnet_b,
        'weight_path': None,  # 缺少 NEU 权重，待上传
        'backbone': 'ResNet-50',
    },
    {
        'name': 'Base-S',
        'dataset': 'NEU-Seg',
        'n_classes': 4,
        'loader': load_base_s,
        'weight_path': None,  # 缺少 NEU 权重，待上传
        'backbone': 'ResNet-18',
    },
    {
        'name': 'A2MS-DefectNet-S',
        'dataset': 'NEU-Seg',
        'n_classes': 4,
        'loader': load_a2ms_defectnet_s,
        'weight_path': None,  # 缺少 NEU 权重，待上传
        'backbone': 'ResNet-18',
    },
    {
        'name': 'STDC1-Seg',
        'dataset': 'NEU-Seg',
        'n_classes': 4,
        'loader': load_stdc1_seg,
        'weight_path': None,  # 缺少 NEU 权重，待上传
        'backbone': 'STDCNet1446',
    },
    {
        'name': 'PP-LiteSeg-B',
        'dataset': 'NEU-Seg',
        'n_classes': 4,
        'loader': load_ppliteseg_b,
        'weight_path': None,  # 缺少 NEU 权重，待上传
        'backbone': 'ResNet-50',
    },
    {
        'name': 'DDRNet23slim',
        'dataset': 'NEU-Seg',
        'n_classes': 4,
        'loader': load_ddrnet23slim,
        'weight_path': None,  # 缺少 NEU 权重，待上传
        'backbone': 'DDRNet-Internal',
    },
    {
        'name': 'BiSeNetV2-L',
        'dataset': 'NEU-Seg',
        'n_classes': 4,
        'loader': load_bisenetv2,
        'weight_path': None,  # 缺少 NEU 权重和代码
        'backbone': 'BiSeNetV2',
    },
    {
        'name': 'Sub-region UNet',
        'dataset': 'NEU-Seg',
        'n_classes': 4,
        'loader': load_subregion_unet,
        'weight_path': os.path.join(NEW_COPY_DIR, "trainedfile", "finalModel_best_newmodel_neu.pkl"),
        'backbone': 'Sub-region-UNet',
    },
    {
        'name': 'PIDNet-S',
        'dataset': 'NEU-Seg',
        'n_classes': 4,
        'loader': load_pidnet_s,
        'weight_path': None,  # 缺少 NEU 权重，待上传
        'backbone': 'PIDNet-Internal',
    },
]


# ============================================================
# 数据加载
# ============================================================
def get_leather_dataloader(batch_size=1, num_workers=4):
    """加载皮革测试集"""
    from datagenerator_yachi import DataGenerator
    from dataAug_new import Compose, ToTensor, TestRescale

    test_transforms = Compose([
        TestRescale(input_hw=(768, 768)),
        ToTensor(),
    ])
    dataset = DataGenerator(txtpath='dataset/pige/test.txt', transformer=test_transforms)
    loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return loader


class NEUSegDataset(torch.utils.data.Dataset):
    """NEU-Seg 测试集加载器（独立实现，不依赖外部 DataGenerator）"""
    def __init__(self, txt_path='dataset/test_neu.txt', input_hw=(200, 200)):
        self.input_hw = input_hw
        self.pairs = []
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    # test_neu.txt: annotation_path image_path
                    ann_rel, img_rel = parts[0], parts[1]
                    self.pairs.append((
                        os.path.join('dataset', ann_rel),
                        os.path.join('dataset', img_rel),
                    ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ann_path, img_path = self.pairs[idx]
        # 读取标注（灰度，0=背景,1,2,3=缺陷）
        mask = Image.open(ann_path).convert('L')
        mask = mask.resize(self.input_hw, Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)

        # 读取图像
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.input_hw, Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))  # HWC -> CHW

        mask = torch.from_numpy(mask)
        binarymask = (mask > 0).long()
        return img, mask, binarymask


def get_neu_dataloader(batch_size=1, num_workers=4):
    """加载 NEU-Seg 测试集"""
    dataset = NEUSegDataset(txt_path='dataset/test_neu.txt', input_hw=(200, 200))
    loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return loader


# ============================================================
# 评估函数
# ============================================================
def evaluate_model(model, dataloader, n_classes, device='cuda'):
    """在数据集上评估模型，返回混淆矩阵"""
    model.eval()
    running_metrics = runningScore(n_classes)

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating", ncols=80):
            images = images.to(device, dtype=torch.float)
            labels = labels.squeeze(1).to(device, dtype=torch.int64)

            outputs = model(images)
            # 所有模型返回 list/tuple，第一个元素为主输出
            if isinstance(outputs, (list, tuple)):
                pred = outputs[0].data.max(1)[1].cpu().numpy()
            else:
                pred = outputs.data.max(1)[1].cpu().numpy()

            gt = labels.data.cpu().numpy()
            running_metrics.update(gt, pred)

    # 计算指标
    score, cls_iu = running_metrics.get_scores()
    metrics = compute_metrics_from_confusion(running_metrics.confusion_matrix)

    return score, metrics, running_metrics.confusion_matrix


# ============================================================
# 数据集配置
# ============================================================
DATASET_CONFIGS = {
    'leather': {
        'name': 'Leather',
        'n_classes': 8,
        'classes': LEATHER_CLASSES,
        'classes_cn': LEATHER_CLASSES_CN,
        'loader_fn': get_leather_dataloader,
        'model_configs': MODEL_CONFIGS,
    },
    'neu': {
        'name': 'NEU-Seg',
        'n_classes': 4,
        'classes': NEU_CLASSES,
        'classes_cn': NEU_CLASSES_CN,
        'loader_fn': get_neu_dataloader,
        'model_configs': NEU_MODEL_CONFIGS,
    },
}


# ============================================================
# 主流程
# ============================================================
def run_evaluation(dataset_key, device, gpu_name):
    """运行单个数据集的评估"""
    ds_cfg = DATASET_CONFIGS[dataset_key]
    ds_name = ds_cfg['name']
    n_classes = ds_cfg['n_classes']
    classes = ds_cfg['classes']
    classes_cn = ds_cfg['classes_cn']
    model_configs = ds_cfg['model_configs']

    print(f"\n{'#'*60}")
    print(f"# 数据集: {ds_name} ({n_classes} 类)")
    print(f"{'#'*60}")

    print(f"加载 {ds_name} 测试集...")
    dataloader = ds_cfg['loader_fn'](batch_size=1, num_workers=4)
    print(f"测试集样本数: {len(dataloader.dataset)}")
    print()

    all_results = []
    errors = []

    for cfg in model_configs:
        name = cfg['name']
        dataset_name = cfg['dataset']
        n_classes = cfg['n_classes']
        weight_path = cfg['weight_path']

        print(f"{'='*60}")
        print(f"模型: {name} | 数据集: {dataset_name}")
        print(f"{'='*60}")

        # 检查权重
        if weight_path is None or not os.path.exists(weight_path):
            reason = "缺少权重，未评估"
            if weight_path is not None:
                reason = f"权重文件不存在: {weight_path}"
            print(f"  [跳过] {reason}")
            errors.append({'model': name, 'dataset': dataset_name, 'error': reason})

            # 记录空结果
            for cls_id, cls_name in classes.items():
                all_results.append({
                    'model': name,
                    'dataset': dataset_name,
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'iou': 'N/A',
                    'dice': 'N/A',
                    'recall': 'N/A',
                    'precision': 'N/A',
                    'script': 'tools/evaluate_class_iou.py',
                    'weight_path': weight_path or 'N/A',
                    'date': time.strftime('%Y-%m-%d'),
                    'verified': 'no',
                })
            all_results.append({
                'model': name,
                'dataset': dataset_name,
                'class_id': -1,
                'class_name': 'mIoU',
                'iou': 'N/A',
                'dice': 'N/A',
                'recall': 'N/A',
                'precision': 'N/A',
                'script': 'tools/evaluate_class_iou.py',
                'weight_path': weight_path or 'N/A',
                'date': time.strftime('%Y-%m-%d'),
                'verified': 'no',
            })
            print()
            continue

        # 加载模型
        try:
            print(f"  加载模型... (权重: {os.path.basename(weight_path)})")
            model = cfg['loader'](weight_path, n_classes=n_classes, device=device)
            model.eval()
            print(f"  模型加载成功")
        except Exception as e:
            err_msg = f"模型加载失败: {e}"
            print(f"  [错误] {err_msg}")
            import traceback
            traceback.print_exc()
            errors.append({'model': name, 'dataset': dataset_name, 'error': err_msg})
            print()
            continue

        # 评估
        try:
            print(f"  开始评估...")
            score, metrics, confusion = evaluate_model(model, dataloader, n_classes, device=device)

            # 打印结果
            print(f"\n  --- 汇总指标 ---")
            for k, v in score.items():
                print(f"  {k}: {v:.4f}")

            class_names = classes
            print(f"\n  --- 各类别指标 ---")
            print(f"  {'类别':<16} {'IoU':<10} {'Dice':<10} {'Recall':<10} {'Precision':<10}")
            print(f"  {'-'*56}")
            for cls_id in range(n_classes):
                cls_name = class_names.get(cls_id, f"class_{cls_id}")
                iou_val = metrics['iou'][cls_id]
                dice_val = metrics['dice'][cls_id]
                recall_val = metrics['recall'][cls_id]
                prec_val = metrics['precision'][cls_id]
                print(f"  {cls_name:<16} {iou_val:<10.4f} {dice_val:<10.4f} {recall_val:<10.4f} {prec_val:<10.4f}")
            print(f"  {'-'*56}")
            print(f"  {'mIoU':<16} {metrics['miou']:<10.4f}")
            print()

            # 记录结果
            for cls_id in range(n_classes):
                cls_name = class_names.get(cls_id, f"class_{cls_id}")
                all_results.append({
                    'model': name,
                    'dataset': dataset_name,
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'iou': f"{metrics['iou'][cls_id]:.4f}",
                    'dice': f"{metrics['dice'][cls_id]:.4f}",
                    'recall': f"{metrics['recall'][cls_id]:.4f}",
                    'precision': f"{metrics['precision'][cls_id]:.4f}",
                    'script': 'tools/evaluate_class_iou.py',
                    'weight_path': weight_path,
                    'date': time.strftime('%Y-%m-%d'),
                    'verified': 'no',
                })
            # mIoU 汇总行
            all_results.append({
                'model': name,
                'dataset': dataset_name,
                'class_id': -1,
                'class_name': 'mIoU',
                'iou': f"{metrics['miou']:.4f}",
                'dice': 'N/A',
                'recall': 'N/A',
                'precision': 'N/A',
                'script': 'tools/evaluate_class_iou.py',
                'weight_path': weight_path,
                'date': time.strftime('%Y-%m-%d'),
                'verified': 'no',
            })

        except Exception as e:
            err_msg = f"评估失败: {e}"
            print(f"  [错误] {err_msg}")
            import traceback
            traceback.print_exc()
            errors.append({'model': name, 'dataset': dataset_name, 'error': err_msg})

        # 释放显存
        del model
        torch.cuda.empty_cache()
        print()

    return all_results, errors, dataloader


def save_results(all_results, errors, model_configs, ds_cfg, dataloader, gpu_name):
    """保存评估结果到 CSV 和 Markdown"""
    ds_name = ds_cfg['name']
    classes = ds_cfg['classes']
    classes_cn = ds_cfg['classes_cn']
    n_classes = ds_cfg['n_classes']

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'experiments', 'results'
    )
    os.makedirs(results_dir, exist_ok=True)

    # 保存 CSV
    csv_path = os.path.join(results_dir, 'class_iou_results.csv')
    fieldnames = [
        'model', 'dataset', 'class_id', 'class_name',
        'iou', 'dice', 'recall', 'precision',
        'script', 'weight_path', 'date', 'verified'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"CSV 已保存到: {csv_path}")

    # 保存汇总 Markdown
    summary_path = os.path.join(results_dir, 'class_iou_summary.md')
    with open(summary_path, 'w') as f:
        f.write("# 类别级 IoU 评估汇总\n\n")
        f.write(f"- 日期: {time.strftime('%Y-%m-%d')}\n")
        f.write(f"- 设备: {gpu_name}\n")
        f.write(f"- 评估脚本: `tools/evaluate_class_iou.py`\n")
        f.write(f"- 数据集: {ds_name} ({len(dataloader.dataset)} 张)\n\n")

        # 类别说明
        f.write("## 类别说明\n\n")
        f.write(f"### {ds_name} ({n_classes} 类含背景)\n\n")
        f.write("| 类别 ID | 英文名 | 中文名 |\n")
        f.write("|---|---|---|\n")
        for cls_id, cls_name in classes.items():
            cn_name = classes_cn[cls_id]
            f.write(f"| {cls_id} | {cls_name} | {cn_name} |\n")

        # 模型对比表 — mIoU
        f.write("\n## 模型对比 (mIoU)\n\n")
        f.write("| 模型 | Backbone | 权重 | mIoU |\n")
        f.write("|---|---|---|---|\n")
        for cfg in model_configs:
            name = cfg['name']
            miou_row = [r for r in all_results if r['model'] == name and r['class_name'] == 'mIoU']
            miou_val = miou_row[0]['iou'] if miou_row else 'N/A'
            weight_status = '有' if cfg['weight_path'] and os.path.exists(cfg['weight_path']) else '缺少'
            f.write(f"| {name} | {cfg['backbone']} | {weight_status} | {miou_val} |\n")

        # 各类别详细对比
        evaluated_models = [cfg['name'] for cfg in model_configs
                           if cfg['weight_path'] and os.path.exists(cfg['weight_path'])]

        if evaluated_models:
            f.write("\n## 各类别 IoU 详细对比\n\n")
            header = "| 类别 |"
            sep = "|---|"
            for m in evaluated_models:
                header += f" {m} |"
                sep += "---|"
            f.write(header + "\n" + sep + "\n")

            for cls_id in range(n_classes):
                cls_name = classes_cn[cls_id]
                row = f"| {cls_name} |"
                for m in evaluated_models:
                    val = [r for r in all_results
                           if r['model'] == m and r['class_id'] == cls_id]
                    if val:
                        row += f" {val[0]['iou']} |"
                    else:
                        row += " N/A |"
                f.write(row + "\n")

            # 各类别 Dice
            f.write("\n## 各类别 Dice 详细对比\n\n")
            header = "| 类别 |"
            sep = "|---|"
            for m in evaluated_models:
                header += f" {m} |"
                sep += "---|"
            f.write(header + "\n" + sep + "\n")

            for cls_id in range(n_classes):
                cls_name = classes_cn[cls_id]
                row = f"| {cls_name} |"
                for m in evaluated_models:
                    val = [r for r in all_results
                           if r['model'] == m and r['class_id'] == cls_id]
                    if val:
                        row += f" {val[0]['dice']} |"
                    else:
                        row += " N/A |"
                f.write(row + "\n")

        # 错误记录
        if errors:
            f.write("\n## 错误/跳过记录\n\n")
            for e in errors:
                f.write(f"- **{e['model']}** ({e['dataset']}): {e['error']}\n")

    print(f"汇总已保存到: {summary_path}")

    # 保存错误记录
    if errors:
        import json
        error_path = os.path.join(results_dir, 'class_iou_errors.json')
        with open(error_path, 'w') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"错误记录已保存到: {error_path}")


def main():
    parser = argparse.ArgumentParser(description='类别级 IoU 评估')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['leather', 'neu', 'all'],
                        help='评估的数据集: leather, neu, all (默认: all)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"日期: {time.strftime('%Y-%m-%d')}")

    datasets_to_run = []
    if args.dataset == 'all':
        datasets_to_run = ['leather', 'neu']
    else:
        datasets_to_run = [args.dataset]

    all_results = []
    all_errors = []

    for ds_key in datasets_to_run:
        ds_cfg = DATASET_CONFIGS[ds_key]
        results, errors, dataloader = run_evaluation(ds_key, device, gpu_name)
        all_results.extend(results)
        all_errors.extend(errors)

        # 保存每个数据集的结果
        save_results(all_results, all_errors, ds_cfg['model_configs'],
                     ds_cfg, dataloader, gpu_name)

    print("\n所有评估完成！")


if __name__ == '__main__':
    main()

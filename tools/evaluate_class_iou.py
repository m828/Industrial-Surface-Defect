#!/usr/bin/env python3
"""Unified per-class IoU evaluator for industrial surface defect segmentation.

This script evaluates one model on one locked test list. It intentionally does
not train, modify model definitions, or write any checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data
from tqdm import tqdm

WORKSPACE = Path("/workspace/Industrial Surface Defect")
REPO_DIR = WORKSPACE / "Industrial-Surface-Defect"
NEW_DIR = WORKSPACE / "new"
NEW_COPY_DIR = WORKSPACE / "new (copy)"

# Put historical code roots on sys.path so model files can import local helpers.
for _path in (str(NEW_DIR), str(NEW_COPY_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

LEATHER_CLASSES = [
    "background",
    "open_wound",
    "scratch",
    "brand_mark",
    "hole",
    "skin_disease",
    "rotten_surface",
    "wart",
]

NEU_CLASSES = ["background", "crazing", "inclusion", "patches"]

MODEL_ALIASES = {
    "base-s": "Base-S",
    "base_s": "Base-S",
    "dsmors18": "Base-S",
    "base-b": "Base-B",
    "base_b": "Base-B",
    "dsmors50": "Base-B",
    "a2ms-defectnet-s": "A2MS-DefectNet-S",
    "a2ms_s": "A2MS-DefectNet-S",
    "a2ms-s": "A2MS-DefectNet-S",
    "dsmors18_ese_adapt_detailloss": "A2MS-DefectNet-S",
    "dsmors18_eSE_adapt_detailloss": "A2MS-DefectNet-S",
    "a2ms-defectnet-b": "A2MS-DefectNet-B",
    "a2ms_b": "A2MS-DefectNet-B",
    "a2ms-b": "A2MS-DefectNet-B",
    "dsmors50_ese_adapt_detailloss": "A2MS-DefectNet-B",
    "dsmors50_eSE_adapt_detailloss": "A2MS-DefectNet-B",
    "stdc-seg": "STDC-Seg",
    "stdc1-seg": "STDC-Seg",
    "stdc1": "STDC-Seg",
    "stdc": "STDC-Seg",
    "ddrnet23slim": "DDRNet23slim",
    "ddrnet": "DDRNet23slim",
    "ddr": "DDRNet23slim",
    "pidnet-s": "PIDNet-S",
    "pidnet_s": "PIDNet-S",
    "pid": "PIDNet-S",
    "pp-liteseg-b": "PP-LiteSeg-B",
    "ppliteseg-b": "PP-LiteSeg-B",
    "ppliteseg_b": "PP-LiteSeg-B",
    "bisenetv1-l": "BiSeNetV1-L",
    "bisenetv1_l": "BiSeNetV1-L",
    "fdsnet": "FDSNet",
    "fds": "FDSNet",
    "letnet": "LETNet",
    "let": "LETNet",
}

DEFAULT_MODEL_FILES = {
    "Base-S": NEW_DIR / "model_dsmo_rs18.py",
    "Base-B": NEW_DIR / "model_dsmo_rs50.py",
    "A2MS-DefectNet-S": NEW_DIR / "model_dsmo_rs18_eSE_adapt_detailloss_822.py",
    "A2MS-DefectNet-B": NEW_DIR / "model_dsmo_rs50_eSE_adapt_detailloss.py",
    "STDC-Seg": NEW_COPY_DIR / "stdc.py",
    "DDRNet23slim": NEW_DIR / "model_ddr.py",
    "PIDNet-S": NEW_DIR / "pid.py",
    "PP-LiteSeg-B": NEW_DIR / "model_dsmo_rs50_ppliteseg.py",
    "BiSeNetV1-L": NEW_DIR / "model_dsmo_rs50_csfcn_yuan.py",
    "FDSNet": REPO_DIR / "third_party" / "FDSNet" / "core" / "models" / "fdsnet.py",
    "LETNet": REPO_DIR / "third_party" / "LETNet" / "Network" / "model" / "LETNet.py",
}

DEFAULT_DATA_ROOTS = {
    "leather": NEW_COPY_DIR / "dataset" / "pige",
    "neu": NEW_COPY_DIR / "dataset",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class SegmentationListDataset(data.Dataset):
    def __init__(
        self,
        dataset: str,
        test_list: Path,
        data_root: Path,
        input_size: Tuple[int, int],
    ) -> None:
        self.dataset = dataset
        self.test_list = test_list
        self.data_root = data_root
        self.input_size = input_size
        self.samples = self._read_samples()

    def _read_samples(self) -> List[Tuple[Path, Path, str, str, int]]:
        samples: List[Tuple[Path, Path, str, str, int]] = []
        with self.test_list.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Malformed line {lineno} in {self.test_list}: {line!r}")
                first, second = parts
                if self.dataset == "neu":
                    label_rel, image_rel = first, second
                else:
                    image_rel, label_rel = first, second
                image_path = self.data_root / image_rel
                label_path = self.data_root / label_rel
                if not image_path.exists():
                    raise FileNotFoundError(f"Missing image at line {lineno}: {image_path}")
                if not label_path.exists():
                    raise FileNotFoundError(f"Missing label at line {lineno}: {label_path}")
                samples.append((image_path, label_path, image_rel, label_rel, lineno))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label_path, image_rel, label_rel, lineno = self.samples[idx]
        height, width = self.input_size

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")
        if image.size != (width, height):
            image = image.resize((width, height), Image.BILINEAR)
        if label.size != (width, height):
            label = label.resize((width, height), Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        label_np = np.asarray(label, dtype=np.int64)
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
        label_tensor = torch.from_numpy(label_np)
        return image_tensor, label_tensor, image_rel, label_rel, lineno


def normalize_model_name(model_name: str) -> str:
    key = model_name.strip()
    if key in DEFAULT_MODEL_FILES:
        return key
    lowered = key.lower()
    return MODEL_ALIASES.get(lowered, MODEL_ALIASES.get(key, key))


def load_module_from_file(model_file: Path, module_tag: str):
    model_file = model_file.resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_file}")
    parent = str(model_file.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(module_tag, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def torch_load_checkpoint(weight_path: Path, device: torch.device):
    try:
        return torch.load(weight_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(weight_path, map_location=device)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(hasattr(v, "shape") for v in checkpoint.values()):
            return checkpoint
    return checkpoint


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_state(model: torch.nn.Module, weight_path: Path, device: torch.device, strict: bool = True) -> None:
    checkpoint = torch_load_checkpoint(weight_path, device)
    state = extract_state_dict(checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
    state = strip_module_prefix(state)
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load weight {weight_path}. This model/weight pair likely needs separate adaptation. {exc}"
        ) from exc


def build_base_b(model_file: Path, num_classes: int, device: torch.device) -> Tuple[torch.nn.Module, str]:
    from model_resnet import resnet50

    module = load_module_from_file(model_file, "eval_base_b_module")
    model = module.DSMONet(num_classes=num_classes, backbone=resnet50()).to(device)
    return model, "DSMONet(num_classes=N, backbone=resnet50())"


def build_base_s(model_file: Path, num_classes: int, device: torch.device) -> Tuple[torch.nn.Module, str]:
    from model_resnet import resnet18

    module = load_module_from_file(model_file, "eval_base_s_module")
    model = module.DSMONet(num_classes=num_classes, backbone=resnet18()).to(device)
    return model, "DSMONet(num_classes=N, backbone=resnet18())"


def build_a2ms_b(model_file: Path, num_classes: int, device: torch.device) -> Tuple[torch.nn.Module, str]:
    from model_resnet import resnet50

    module = load_module_from_file(model_file, "eval_a2ms_b_module")
    model = module.DSMONet(num_classes=num_classes, backbone=resnet50()).to(device)
    return model, "DSMONet(num_classes=N, backbone=resnet50())"


def build_a2ms_s(model_file: Path, num_classes: int, device: torch.device) -> Tuple[torch.nn.Module, str]:
    from model_resnet import resnet18

    module = load_module_from_file(model_file, "eval_a2ms_s_module")
    model = module.DSMONet(num_classes=num_classes, backbone=resnet18()).to(device)
    return model, "DSMONet(num_classes=N, backbone=resnet18())"


def build_stdc1(model_file: Path, num_classes: int, device: torch.device) -> Tuple[torch.nn.Module, str]:
    module = load_module_from_file(model_file, "eval_stdc_module")
    model = module.BiSeNet(
        backbone="STDCNet1446",
        n_classes=num_classes,
        pretrain_model="",
        use_boundary_8=True,
    ).to(device)
    return model, "BiSeNet(backbone='STDCNet1446', n_classes=N, use_boundary_8=True)"


def build_ddrnet(model_file: Path, num_classes: int, device: torch.device) -> Tuple[torch.nn.Module, str]:
    module = load_module_from_file(model_file, "eval_ddr_module")
    if hasattr(module, "DualResNet_imagenet"):
        model = module.DualResNet_imagenet(num_classes=num_classes).to(device)
        return model, "DualResNet_imagenet(num_classes=N)"
    raise AttributeError(f"{model_file} does not expose DualResNet_imagenet")


def build_pidnet(model_file: Path, num_classes: int, device: torch.device) -> Tuple[torch.nn.Module, str]:
    module = load_module_from_file(model_file, "eval_pid_module")
    model = module.PIDNet(
        m=2,
        n=3,
        num_classes=num_classes,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=True,
    ).to(device)
    return model, "PIDNet(m=2, n=3, planes=32, ppm_planes=96, head_planes=128, augment=True)"


def build_ppliteseg_b(model_file: Path, num_classes: int, device: torch.device) -> Tuple[torch.nn.Module, str]:
    from model_resnet import resnet50

    module = load_module_from_file(model_file, "eval_ppliteseg_b_module")
    model = module.DSMONet(num_classes=num_classes, backbone=resnet50()).to(device)
    return model, "DSMONet(PP-LiteSeg variant, num_classes=N, backbone=resnet50())"


def build_bisenetv1_l(model_file: Path, num_classes: int, device: torch.device) -> Tuple[torch.nn.Module, str]:
    from model_resnet import resnet50

    module = load_module_from_file(model_file, "eval_bisenetv1_l_module")
    model = module.DSMONet(num_classes=num_classes, backbone=resnet50()).to(device)
    return model, "DSMONet(CSFCN/BiSeNetV1-L candidate, num_classes=N, backbone=resnet50())"


def _install_mmcv_shim() -> None:
    """Minimal mmcv.cnn shim for FDSNet's gcblock (constant_init/kaiming_init only).
    mmcv is intentionally not installed (environment conflict); see
    experiments/baseline_comparison/industrial_baseline_audit.md."""
    if "mmcv.cnn" in sys.modules:
        return
    import types

    mmcv = types.ModuleType("mmcv")
    mmcv_cnn = types.ModuleType("mmcv.cnn")

    def constant_init(m, val=0):
        if hasattr(m, "weight") and m.weight is not None:
            torch.nn.init.constant_(m.weight, val)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias, val)

    def kaiming_init(m, mode="fan_out", nonlinearity="relu", **kw):
        torch.nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

    mmcv_cnn.constant_init = constant_init
    mmcv_cnn.kaiming_init = kaiming_init
    mmcv.cnn = mmcv_cnn
    sys.modules.setdefault("mmcv", mmcv)
    sys.modules.setdefault("mmcv.cnn", mmcv_cnn)


def build_fdsnet(model_file: Path, num_classes: int, device: torch.device) -> Tuple[torch.nn.Module, str]:
    # The repo's core/__init__ is broken upstream (missing core.nn.jpu); load the
    # model file directly under a stub parent package so its relative imports work.
    import types

    _install_mmcv_shim()
    pkg = types.ModuleType("fdsnet_models")
    pkg.__path__ = [str(model_file.parent)]
    sys.modules.setdefault("fdsnet_models", pkg)
    module = load_module_from_file(model_file, "fdsnet_models.fdsnet")
    model = module.FDSNet(num_classes=num_classes, aux=True).to(device)
    return model, "FDSNet(num_classes=N, aux=True) [ICASSP 2022]"


class _PadCropWrapper(torch.nn.Module):
    """Pad HxW input to pad_to x pad_to (right/bottom, zeros), run inner model,
    crop logits back to HxW. Used for LETNet (/16-divisible input constraint)."""

    def __init__(self, inner: torch.nn.Module, pad_to: int = 208):
        super().__init__()
        self.inner = inner
        self.pad_to = pad_to

    def load_state_dict(self, state_dict, strict: bool = True):
        # Training checkpoints store the RAW inner model's keys (no "inner."
        # prefix); accept both layouts so strict loading works either way.
        if not any(k.startswith("inner.") for k in state_dict.keys()):
            state_dict = {f"inner.{k}": v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.pad_to - w, 0, self.pad_to - h))
        out = self.inner(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out[..., :h, :w]


def build_letnet(model_file: Path, num_classes: int, device: torch.device) -> Tuple[torch.nn.Module, str]:
    # LETNet imports "from module.transformer import ..." -> needs Network/model on path.
    network_dir = model_file.parent.parent  # third_party/LETNet/Network
    for _p in (str(network_dir), str(network_dir / "model")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    module = load_module_from_file(model_file, "eval_letnet_module")
    inner = module.LETNet(classes=num_classes)
    model = _PadCropWrapper(inner, pad_to=208).to(device)
    return model, "LETNet(classes=N) with 200->208 pad / 208->200 crop wrapper"


MODEL_BUILDERS: Dict[str, Callable[[Path, int, torch.device], Tuple[torch.nn.Module, str]]] = {
    "Base-S": build_base_s,
    "Base-B": build_base_b,
    "A2MS-DefectNet-S": build_a2ms_s,
    "A2MS-DefectNet-B": build_a2ms_b,
    "STDC-Seg": build_stdc1,
    "DDRNet23slim": build_ddrnet,
    "PIDNet-S": build_pidnet,
    "PP-LiteSeg-B": build_ppliteseg_b,
    "BiSeNetV1-L": build_bisenetv1_l,
    "FDSNet": build_fdsnet,
    "LETNet": build_letnet,
}


def confusion_matrix_update(confusion: np.ndarray, label: np.ndarray, pred: np.ndarray, num_classes: int) -> None:
    mask = (label >= 0) & (label < num_classes)
    hist = np.bincount(
        num_classes * label[mask].astype(np.int64) + pred[mask].astype(np.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    confusion += hist


def compute_metrics(confusion: np.ndarray) -> Dict[str, np.ndarray | float]:
    tp = np.diag(confusion).astype(np.float64)
    row_sum = confusion.sum(axis=1).astype(np.float64)
    col_sum = confusion.sum(axis=0).astype(np.float64)
    denom_iou = row_sum + col_sum - tp

    iou = np.divide(tp, denom_iou, out=np.full_like(tp, np.nan), where=denom_iou > 0)
    dice_denom = row_sum + col_sum
    dice = np.divide(2 * tp, dice_denom, out=np.full_like(tp, np.nan), where=dice_denom > 0)
    recall = np.divide(tp, row_sum, out=np.full_like(tp, np.nan), where=row_sum > 0)
    precision = np.divide(tp, col_sum, out=np.full_like(tp, np.nan), where=col_sum > 0)
    miou = float(np.nanmean(iou))
    return {
        "iou": iou,
        "dice": dice,
        "recall": recall,
        "precision": precision,
        "miou": miou,
    }


def resolve_class_names(dataset: str, num_classes: int, class_names: Optional[Sequence[str]]) -> List[str]:
    if class_names:
        if len(class_names) != num_classes:
            raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}")
        return list(class_names)
    defaults = LEATHER_CLASSES if dataset == "leather" else NEU_CLASSES
    if len(defaults) == num_classes:
        return list(defaults)
    return [f"class_{i}" for i in range(num_classes)]


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    dataset_key = args.dataset.lower()
    if dataset_key not in ("leather", "neu"):
        raise ValueError("--dataset must be leather or neu")

    model_name = normalize_model_name(args.model_name)
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model name: {args.model_name}. Supported: {sorted(MODEL_BUILDERS)}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else "N/A"

    test_list = Path(args.test_list).resolve()
    if not test_list.exists():
        raise FileNotFoundError(f"Test list does not exist: {test_list}")

    data_root = Path(args.data_root).resolve() if args.data_root else DEFAULT_DATA_ROOTS[dataset_key]
    model_file = Path(args.model_file).resolve() if args.model_file else DEFAULT_MODEL_FILES[model_name]
    weight_path = Path(args.weight).resolve()
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight does not exist: {weight_path}")

    class_names = resolve_class_names(dataset_key, args.num_classes, args.class_names)
    input_size = (int(args.input_size[0]), int(args.input_size[1]))
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("# Unified Class IoU Evaluation")
    print(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"dataset: {dataset_key}")
    print(f"test_list: {test_list}")
    print(f"data_root: {data_root}")
    print(f"model_name: {model_name}")
    print(f"model_file: {model_file}")
    print(f"weight_path: {weight_path}")
    print(f"num_classes: {args.num_classes}")
    print(f"input_size: {input_size[0]}x{input_size[1]}")
    print("preprocess: PIL resize(TestRescale equivalent) + ToTensor(/255), no ImageNet Normalize")
    print("miou_includes_background: true")
    print(f"device: {device}")
    print(f"gpu: {gpu_name}")

    dataset = SegmentationListDataset(dataset_key, test_list, data_root, input_size)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"test_samples: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("No valid samples in test list")

    model, model_entry = MODEL_BUILDERS[model_name](model_file, args.num_classes, device)
    print(f"model_entry: {model_entry}")
    load_state(model, weight_path, device, strict=not args.non_strict)
    print("weight_load: success")

    model.eval()
    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)

    with torch.no_grad():
        for images, labels, _image_rel, _label_rel, _lineno in tqdm(loader, desc="Evaluating", ncols=80):
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            gt = labels.detach().cpu().numpy()
            for gt_i, pred_i in zip(gt, pred):
                confusion_matrix_update(confusion, gt_i, pred_i, args.num_classes)

    metrics = compute_metrics(confusion)
    print("\nconfusion_matrix:")
    for row in confusion.astype(np.int64):
        print("[" + ", ".join(str(int(v)) for v in row) + "]")

    print("\nper_class_metrics:")
    print("class_id,class_name,iou,dice,precision,recall")
    rows = []
    for class_id, class_name in enumerate(class_names):
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "iou": float(metrics["iou"][class_id]),
            "dice": float(metrics["dice"][class_id]),
            "precision": float(metrics["precision"][class_id]),
            "recall": float(metrics["recall"][class_id]),
        }
        rows.append(row)
        print(
            f"{class_id},{class_name},{row['iou']:.6f},{row['dice']:.6f},"
            f"{row['precision']:.6f},{row['recall']:.6f}"
        )
    print(f"\nmIoU: {metrics['miou']:.6f}")

    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_key,
        "test_list": str(test_list),
        "data_root": str(data_root),
        "test_samples": len(dataset),
        "model_name": model_name,
        "model_file": str(model_file),
        "model_entry": model_entry,
        "weight_path": str(weight_path),
        "num_classes": args.num_classes,
        "input_size": [input_size[0], input_size[1]],
        "miou_includes_background": True,
        "preprocess": "TestRescale equivalent + ToTensor(/255), no ImageNet Normalize",
        "device": str(device),
        "gpu": gpu_name,
        "output_csv": str(output_csv),
    }

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "row_type",
            "dataset",
            "model_name",
            "class_id",
            "class_name",
            "iou",
            "dice",
            "precision",
            "recall",
            "miou",
            "test_samples",
            "includes_background",
            "weight_path",
            "model_file",
            "model_entry",
            "date",
            "device",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "row_type": "class",
                "dataset": dataset_key,
                "model_name": model_name,
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "iou": f"{row['iou']:.6f}",
                "dice": f"{row['dice']:.6f}",
                "precision": f"{row['precision']:.6f}",
                "recall": f"{row['recall']:.6f}",
                "miou": "",
                "test_samples": len(dataset),
                "includes_background": "true",
                "weight_path": str(weight_path),
                "model_file": str(model_file),
                "model_entry": model_entry,
                "date": metadata["date"],
                "device": str(device),
            })
        writer.writerow({
            "row_type": "summary",
            "dataset": dataset_key,
            "model_name": model_name,
            "class_id": -1,
            "class_name": "mIoU",
            "iou": "",
            "dice": "",
            "precision": "",
            "recall": "",
            "miou": f"{metrics['miou']:.6f}",
            "test_samples": len(dataset),
            "includes_background": "true",
            "weight_path": str(weight_path),
            "model_file": str(model_file),
            "model_entry": model_entry,
            "date": metadata["date"],
            "device": str(device),
        })
        writer.writerow({
            "row_type": "metadata",
            "dataset": dataset_key,
            "model_name": model_name,
            "class_id": "",
            "class_name": "confusion_matrix_json",
            "iou": json.dumps(confusion.astype(int).tolist()),
            "dice": "",
            "precision": "",
            "recall": "",
            "miou": "",
            "test_samples": len(dataset),
            "includes_background": "true",
            "weight_path": str(weight_path),
            "model_file": str(model_file),
            "model_entry": model_entry,
            "date": metadata["date"],
            "device": str(device),
        })

    print(f"\noutput_csv: {output_csv}")
    return {
        "metadata": metadata,
        "class_rows": rows,
        "miou": metrics["miou"],
        "confusion": confusion.astype(int).tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified per-class IoU evaluator")
    parser.add_argument("--dataset", required=True, choices=["leather", "neu"], help="Dataset key")
    parser.add_argument("--test-list", "--test_txt", dest="test_list", required=True, help="Test list path")
    parser.add_argument("--data-root", dest="data_root", default=None, help="Dataset root for relative paths")
    parser.add_argument("--model-name", "--model_type", dest="model_name", required=True, help="Model name or alias")
    parser.add_argument("--model-file", dest="model_file", default=None, help="Model definition .py file")
    parser.add_argument("--weight", "--weight_path", dest="weight", required=True, help="Checkpoint path")
    parser.add_argument("--num-classes", "--num_classes", dest="num_classes", required=True, type=int)
    parser.add_argument("--input-size", "--input_size", dest="input_size", nargs=2, required=True, type=int, metavar=("H", "W"))
    parser.add_argument("--class-names", "--class_names", dest="class_names", nargs="*", default=None)
    parser.add_argument("--output-csv", dest="output_csv", default=None, help="CSV output path")
    parser.add_argument("--output_dir", dest="output_dir", default=None, help="Compatibility output dir; writes class_iou.csv inside")
    parser.add_argument("--save-log", dest="save_log", default=None, help="Optional log path")
    parser.add_argument("--batch-size", dest="batch_size", default=1, type=int)
    parser.add_argument("--num-workers", dest="num_workers", default=4, type=int)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--non-strict", action="store_true", help="Load state_dict with strict=False")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.output_csv is None:
        if args.output_dir is None:
            parser.error("Either --output-csv or --output_dir is required")
        args.output_csv = str(Path(args.output_dir) / "class_iou.csv")

    log_path = Path(args.save_log).resolve() if args.save_log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            tee_out = Tee(sys.stdout, log_file)
            tee_err = Tee(sys.stderr, log_file)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                try:
                    evaluate(args)
                    print(f"save_log: {log_path}")
                    return 0
                except Exception:
                    traceback.print_exc()
                    return 1
    try:
        evaluate(args)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

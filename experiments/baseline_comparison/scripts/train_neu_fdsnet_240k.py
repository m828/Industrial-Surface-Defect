# Protocol-compliant NEU-Seg baseline training: FDSNet (ICASSP 2022, industrial defect).
# Source: third_party/FDSNet @ ced4d0d (github.com/jianzhang96/fdsnet).
# Protocolized identically to the other NEU baselines:
#   * NO val/test loader, NO model selection; checkpoint ONLY at final iteration
#   * Adam lr=1e-4 CONSTANT (no schedule), wd=2e-6, seed 1337, from scratch
#   * batch 16, input 200x200, 4 classes
#   * checkpoint dict carries "model_state" and "iter" keys (unified evaluator contract)
# Chain-habit loss (mirrors FDSNet's FDSNetLoss + train.py weighting):
#   loss = 1.0 * OHEM-CE(main, label)          [OhemCrossEntropy2d: thresh 0.7, min_kept 1e5]
#        + 0.5 * BCE(edge_aux, edge_gt)        [edge GT generated online: morphological
#                                               boundary of the multi-class label map,
#                                               (dilate3 != erode3); matches the binarized
#                                               AuxiliaryGT convention `(target_edge > 0)`]
#        + 0.5 * BCE(softmax(se_aux), se_gt)   [image-level presence vector of classes 1..3]
# Environment notes:
#   * core/__init__ of the repo is broken upstream (missing core.nn.jpu); the model file is
#     loaded directly via importlib with a stub parent package.
#   * gcblock.py imports mmcv.cnn.constant_init/kaiming_init; mmcv is NOT installed
#     (SeaFormer was excluded for the same conflict). A minimal init-equivalent shim is
#     injected into sys.modules (documented in industrial_baseline_audit.md).

import argparse
import importlib.util
import json
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils import data

WORKSPACE = Path("/workspace/Industrial Surface Defect")
NEW_COPY_DIR = WORKSPACE / "new (copy)"
REPO_DIR = WORKSPACE / "Industrial-Surface-Defect"
FDSNET_DIR = REPO_DIR / "third_party" / "FDSNet"

for _p in (str(NEW_COPY_DIR),):  # datagenerator_neu, dataAug_new
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datagenerator_neu import DataGenerator  # noqa: E402
from dataAug_new import Compose, Transforms_PIL, ToTensor  # noqa: E402


def _install_mmcv_shim():
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


def build_model(n_classes, device):
    _install_mmcv_shim()
    pkg = types.ModuleType("fdsnet_models")
    pkg.__path__ = [str(FDSNET_DIR / "core" / "models")]
    sys.modules.setdefault("fdsnet_models", pkg)
    spec = importlib.util.spec_from_file_location(
        "fdsnet_models.fdsnet", FDSNET_DIR / "core" / "models" / "fdsnet.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fdsnet_models.fdsnet"] = module
    spec.loader.exec_module(module)
    model = module.FDSNet(num_classes=n_classes, aux=True).to(device)
    return model


class OhemCrossEntropy2d(torch.nn.Module):
    """Verbatim semantics of FDSNet core/utils/loss.py OhemCrossEntropy2d
    (thresh=0.7, min_kept=100000, plain CE, no class weights)."""

    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)
        if num_valid > 0 and self.min_kept <= num_valid:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long, device=pred.device)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)
        return self.criterion(pred, target)


def make_edge_gt(labels):
    """Binary boundary of the multi-class label map: (dilate3 != erode3).
    labels: [B, H, W] long on device -> [B, 1, H, W] float."""
    lbl = labels.unsqueeze(1).float()
    dil = F.max_pool2d(lbl, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-lbl, kernel_size=3, stride=1, padding=1)
    return (dil != ero).float()


def make_se_gt(labels, n_classes):
    """Image-level presence vector for defect classes 1..n-1 -> [B, n-1] float."""
    b = labels.shape[0]
    onehot = F.one_hot(labels.clamp_min(0), num_classes=n_classes)  # [B,H,W,C]
    present = (onehot.sum(dim=(1, 2)) > 0)[:, 1:]  # skip background
    return present.float().view(b, n_classes - 1)


def _resolve(path_like):
    return Path(path_like).expanduser().resolve()


def _assert_safe_checkpoint_path(path):
    path = _resolve(path_like=path)
    if "model_savePath" in str(path):
        raise RuntimeError("Refusing to save checkpoint under historical model_savePath*: {}".format(path))
    if path.exists():
        raise FileExistsError("Refusing to overwrite existing checkpoint: {}".format(path))
    return path


def train(cfg, logger):
    seed = int(cfg["training"].get("seed", 1337))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    train_list = str(_resolve(data_cfg["train_list"]))
    data_dir = _resolve(data_cfg.get("data_dir", NEW_COPY_DIR))
    n_classes = int(data_cfg.get("num_classes", 4))
    input_hw = tuple(data_cfg.get("input_hw", [200, 200]))

    train_iters = int(cfg["training"]["train_iters"])
    batch_size = int(cfg["training"]["batch_size"])

    experiment_cfg = cfg["experiment"]
    run_id = experiment_cfg.get("run_id") or experiment_cfg.get("run_prefix", "neu_fdsnet_240k")
    checkpoint_dir = _resolve(experiment_cfg["checkpoint_dir"])
    result_dir = _resolve(experiment_cfg["result_dir"])
    for d in (checkpoint_dir, result_dir):
        if "model_savePath" in str(d):
            raise RuntimeError("Refusing historical checkpoint directory: {}".format(d))
        d.mkdir(parents=True, exist_ok=True)

    # DataGenerator opens files via the hardcoded 'dataset/' prefix -> cwd must be data_dir.
    os.chdir(str(data_dir))

    train_transforms = Compose([Transforms_PIL(input_hw=input_hw), ToTensor()])
    train_dataset = DataGenerator(txtpath=train_list, transformer=train_transforms)
    trainloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=int(cfg["training"]["n_workers"]),
        shuffle=True,
        drop_last=True,
    )
    logger.info("Train list: {} ({} samples, {} iters/epoch at bs{})".format(
        train_list, len(train_dataset), len(trainloader), batch_size))
    logger.info("Classes: {} | input: {}x{} | seed: {}".format(n_classes, input_hw[0], input_hw[1], seed))
    logger.info("Protocol: NO val/test loader constructed; NO model selection; constant LR; "
                "checkpoint only at final iter {}.".format(train_iters))

    model = build_model(n_classes, device)
    logger.info("Model: FDSNet(num_classes={}, aux=True) from {}".format(n_classes, FDSNET_DIR))

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=2.0e-6)
    logger.info("Using optimizer Adam(lr=1e-4 const, wd=2e-6) [protocol]")

    ohem = OhemCrossEntropy2d(ignore_index=-1, thresh=0.7, min_kept=100000).to(device)
    bce = torch.nn.BCELoss()
    loss_weights = [1.0, 0.5, 0.5]  # FDSNet chain habit (train.py defaults)
    logger.info("Loss: [OHEM-CE main, BCE edge, BCE semantic] x {}".format(loss_weights))

    metrics_path = result_dir / "{}_train_metrics.jsonl".format(run_id)
    if metrics_path.exists():
        raise FileExistsError("Refusing to overwrite existing metrics: {}".format(metrics_path))

    time_meter = []
    t_start = time.time()
    step = 0
    done = False

    while not done:
        for images, labels, label1 in trainloader:
            step += 1
            iter_t0 = time.time()
            model.train()
            images = images.to(device, dtype=torch.float)
            labels = labels.squeeze(1).to(device, dtype=torch.int64)

            optimizer.zero_grad()
            out_main, out_edge, out_se = model(images)

            edge_gt = make_edge_gt(labels)
            se_gt = make_se_gt(labels, n_classes)

            loss_main = ohem(out_main, labels)
            loss_edge = bce(out_edge.squeeze(1), edge_gt.squeeze(1))
            loss_se = bce(F.softmax(out_se, dim=1), se_gt)
            losses = [loss_main * loss_weights[0],
                      loss_edge * loss_weights[1],
                      loss_se * loss_weights[2]]
            loss = sum(losses)

            loss.backward()
            optimizer.step()

            time_meter.append(time.time() - iter_t0)

            if step % int(cfg["training"]["print_interval"]) == 0 or step == train_iters:
                lr_now = optimizer.param_groups[0]["lr"]
                avg_t = sum(time_meter) / max(1, len(time_meter))
                time_meter = []
                msg = "Iter [{:d}/{:d}]  Loss: {:.4f}  comps: {}  LR: {:.2e}  Time/Iter: {:.4f}".format(
                    step, train_iters, loss.item(),
                    [round(l.item(), 6) for l in losses], lr_now, avg_t)
                print(msg)
                logger.info(msg)
                with open(metrics_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps({
                        "iter": step, "loss": loss.item(),
                        "comps": [l.item() for l in losses],
                        "lr": lr_now, "time_per_iter": avg_t,
                        "wall_time": time.time() - t_start,
                    }) + "\n")

            if step == train_iters:
                state = {
                    "iter": step,
                    "epoch": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "run_id": run_id,
                    "note": "final-iteration checkpoint (protocol); NOT test-selected",
                }
                save_path = _assert_safe_checkpoint_path(
                    checkpoint_dir / "{}_iter{:06d}.pkl".format(run_id, step))
                torch.save(state, str(save_path))
                logger.info("Saved final checkpoint: {}".format(save_path))

                summary = {
                    "run_id": run_id,
                    "model": "FDSNet (ICASSP 2022, aux=True)",
                    "model_file": str(FDSNET_DIR / "core" / "models" / "fdsnet.py"),
                    "train_list": train_list,
                    "train_samples": len(train_dataset),
                    "train_iters": train_iters,
                    "final_checkpoint_path": str(save_path),
                    "wall_time_seconds": time.time() - t_start,
                    "test_during_training": "none (protocol)",
                }
                summary_path = result_dir / "{}_train_summary.json".format(run_id)
                if summary_path.exists():
                    raise FileExistsError("Refusing to overwrite existing summary: {}".format(summary_path))
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                logger.info("Saved training summary: {}".format(summary_path))
                done = True
                break


if __name__ == "__main__":
    import logging
    parser = argparse.ArgumentParser(description="Protocol-compliant NEU FDSNet baseline training")
    parser.add_argument("--config", nargs="?", type=str, required=True, help="Configuration yaml")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    log_dir = _resolve(cfg["experiment"]["log_dir"])
    if "model_savePath" in str(log_dir):
        raise RuntimeError("Refusing historical directory: {}".format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_neu_fdsnet_240k")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_dir / "train.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    logger.info("Let the games begin (protocol baseline: FDSNet NEU)")
    logger.info("Config: {}".format(args.config))

    train(cfg, logger)

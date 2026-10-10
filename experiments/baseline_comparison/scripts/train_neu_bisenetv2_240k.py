# Protocol-compliant NEU-Seg baseline training: BiSeNetV2 (Yu et al. IJCV 2021, real-time).
# Source: third_party/BiSeNetV2 (CoinCheung/BiSeNet @ 6b4b67a), lib/models/bisenetv2.py.
# Protocolized identically to the other NEU baselines:
#   * NO val/test loader, NO model selection; checkpoint ONLY at final iteration
#   * Adam lr=1e-4 CONSTANT (no schedule), wd=2e-6, seed 1337, FROM SCRATCH
#     (BiSeNetV2.load_pretrain is disabled: it would download backbone_v2.pth)
#   * batch 16, input 200x200, 4 classes
#   * checkpoint dict carries "model_state" and "iter" keys (unified evaluator contract)
# Input handling: BGALayer requires (/32 branch) x4 == (/8 detail), so the input must be
# /32-divisible; images are padded 200->224 (zeros) and labels 200->224 (ignore value
# 255), making every head's output exactly 224x224. Evaluation uses the registered
# _PadCropWrapper(pad_to=224), same convention as LETNet (208).
# Chain-habit loss (mirrors the upstream repo's ohem_ce_loss on every head):
#   loss = sum_i OHEM-CE(head_i, label), weight 1.0 each, 5 heads, ignore 255.

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils import data

WORKSPACE = Path("/workspace/Industrial Surface Defect")
NEW_COPY_DIR = WORKSPACE / "new (copy)"
REPO_DIR = WORKSPACE / "Industrial-Surface-Defect"
MODEL_FILE = REPO_DIR / "third_party" / "BiSeNetV2" / "lib" / "models" / "bisenetv2.py"
PAD_TO = 224
IGNORE = 255

for _p in (str(NEW_COPY_DIR),):  # datagenerator_neu, dataAug_new
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datagenerator_neu import DataGenerator  # noqa: E402
from dataAug_new import Compose, Transforms_PIL, ToTensor  # noqa: E402


class OhemCrossEntropy2d(torch.nn.Module):
    """Same semantics as the FDSNet baseline's OHEM (thresh=0.7, min_kept=100000,
    plain CE, no class weights) -> consistent with CoinCheung's OhemCELoss habit."""

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


def build_model(n_classes, device):
    spec = importlib.util.spec_from_file_location("train_bisenetv2_module", MODEL_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # From-scratch protocol: the upstream __init__ -> init_weights -> load_pretrain
    # chain would download backbone_v2.pth from GitHub; replace with a no-op.
    module.BiSeNetV2.load_pretrain = lambda self: None
    model = module.BiSeNetV2(n_classes, aux_mode="train").to(device)
    return model


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
    run_id = experiment_cfg.get("run_id") or experiment_cfg.get("run_prefix", "neu_bisenetv2_240k")
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
    logger.info("Model: BiSeNetV2(n_classes={}, aux_mode=train, load_pretrain DISABLED) from {}".format(
        n_classes, MODEL_FILE))

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=2.0e-6)
    logger.info("Using optimizer Adam(lr=1e-4 const, wd=2e-6) [protocol]")

    ohem = OhemCrossEntropy2d(ignore_index=IGNORE, thresh=0.7, min_kept=100000).to(device)
    logger.info("Loss: OHEM-CE x5 heads, weight 1.0 each, ignore_index=%d (pad region) [chain habit]" % IGNORE)

    metrics_path = result_dir / "{}_train_metrics.jsonl".format(run_id)
    if metrics_path.exists():
        raise FileExistsError("Refusing to overwrite existing metrics: {}".format(metrics_path))

    pad_img = (0, PAD_TO - input_hw[1], 0, PAD_TO - input_hw[0])
    time_meter = []
    t_start = time.time()
    step = 0
    done = False

    while not done:
        for images, labels, label1 in trainloader:
            step += 1
            iter_t0 = time.time()
            model.train()
            images = F.pad(images.to(device, dtype=torch.float), pad_img)
            labels = F.pad(labels.squeeze(1).to(device, dtype=torch.int64), pad_img, value=IGNORE)

            optimizer.zero_grad()
            logits = model(images)  # main, aux2, aux3, aux4, aux5_4 -- all exact 224x224
            if any(lg.shape[-2:] != labels.shape[-2:] for lg in logits):
                raise RuntimeError("unexpected head size at pad_to={}: {}".format(
                    PAD_TO, [tuple(lg.shape) for lg in logits]))
            losses = [ohem(lg, labels) for lg in logits]
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
                    "model": "BiSeNetV2 (CoinCheung/BiSeNet @6b4b67a, aux_mode=train, from scratch)",
                    "model_file": str(MODEL_FILE),
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
    parser = argparse.ArgumentParser(description="Protocol-compliant NEU BiSeNetV2 baseline training")
    parser.add_argument("--config", nargs="?", type=str, required=True, help="Configuration yaml")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    log_dir = _resolve(cfg["experiment"]["log_dir"])
    if "model_savePath" in str(log_dir):
        raise RuntimeError("Refusing historical directory: {}".format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_neu_bisenetv2_240k")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_dir / "train.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    logger.info("Let the games begin (protocol baseline: BiSeNetV2 NEU)")
    logger.info("Config: {}".format(args.config))

    train(cfg, logger)

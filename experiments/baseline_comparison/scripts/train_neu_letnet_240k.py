# Protocol-compliant NEU-Seg baseline training: LETNet (lightweight real-time).
# Source: third_party/LETNet @ faaa065 + local NEU adaptation (see
# results/letnet_adaptation_log.md: 4 upstream blocker fixes + neuseg dataset adapter).
# This trainer replaces LETNet's native recipe (Adam 5e-4, warmpoly, epoch-based,
# in-training val) with the unified baseline protocol:
#   * Adam lr=1e-4 CONSTANT, wd=2e-6, seed 1337, from scratch, batch 16
#   * 240k iterations, NO val/test during training, checkpoint ONLY at final iter
#   * checkpoint dict carries "model_state" and "iter" keys (unified evaluator contract)
# Input handling (per adaptation log): LETNet requires /16-divisible input; NEU 200x200
# is padded to 208x208 (image pad 0, label pad 255) INSIDE the training loop; loss is
# computed on the full 208x208 output against the padded label with ignore_index=255.
# At evaluation the registered wrapper crops the 208x208 output back to 200x200.
# Loss (chain habit): weighted CrossEntropy (class weights from NeuSegTrainInform
# histogram over train_neu.txt: [1.4646, 8.7251, 6.2481, 8.4495]), ignore 255.

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
LETNET_DIR = REPO_DIR / "third_party" / "LETNet" / "Network"
LETNET_MODEL_FILE = LETNET_DIR / "model" / "LETNet.py"

for _p in (str(NEW_COPY_DIR),):  # datagenerator_neu, dataAug_new
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datagenerator_neu import DataGenerator  # noqa: E402
from dataAug_new import Compose, Transforms_PIL, ToTensor  # noqa: E402

CLASS_WEIGHTS = [1.4646, 8.7251, 6.2481, 8.4495]  # NeuSegTrainInform (train_neu.txt)
PAD_TO = 208  # 200 -> 208 (right/bottom); 208 / 16 = 13


def build_model(n_classes, device):
    for _p in (str(LETNET_DIR), str(LETNET_DIR / "model")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    spec = importlib.util.spec_from_file_location("letnet_model", str(LETNET_MODEL_FILE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.LETNet(classes=n_classes).to(device)
    return model


def pad_image_label(images, labels, pad_to):
    """images [B,3,H,W] float; labels [B,H,W] long. Pad right/bottom to pad_to."""
    h, w = images.shape[-2:]
    ph, pw = pad_to - h, pad_to - w
    images = F.pad(images, (0, pw, 0, ph), value=0.0)
    labels = F.pad(labels, (0, pw, 0, ph), value=255)
    return images, labels


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
    run_id = experiment_cfg.get("run_id") or experiment_cfg.get("run_prefix", "neu_letnet_240k")
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
    logger.info("Classes: {} | input: {}x{} (padded to {}x{}) | seed: {}".format(
        n_classes, input_hw[0], input_hw[1], PAD_TO, PAD_TO, seed))
    logger.info("Protocol: NO val/test loader constructed; NO model selection; constant LR; "
                "checkpoint only at final iter {}.".format(train_iters))

    model = build_model(n_classes, device)
    logger.info("Model: LETNet(classes={}) from {}".format(n_classes, LETNET_MODEL_FILE))

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4, weight_decay=2.0e-6)
    logger.info("Using optimizer Adam(lr=1e-4 const, wd=2e-6) [protocol]")

    ce = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float, device=device),
        ignore_index=255)
    logger.info("Loss: weighted CE {} ignore=255 (chain habit)".format(CLASS_WEIGHTS))

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

            images, labels = pad_image_label(images, labels, PAD_TO)

            optimizer.zero_grad()
            output = model(images)  # [B, C, 208, 208]
            loss = ce(output, labels)

            loss.backward()
            optimizer.step()

            time_meter.append(time.time() - iter_t0)

            if step % int(cfg["training"]["print_interval"]) == 0 or step == train_iters:
                lr_now = optimizer.param_groups[0]["lr"]
                avg_t = sum(time_meter) / max(1, len(time_meter))
                time_meter = []
                msg = "Iter [{:d}/{:d}]  Loss: {:.4f}  LR: {:.2e}  Time/Iter: {:.4f}".format(
                    step, train_iters, loss.item(), lr_now, avg_t)
                print(msg)
                logger.info(msg)
                with open(metrics_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps({
                        "iter": step, "loss": loss.item(),
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
                    "model": "LETNet(classes=4), 208x208 pad input",
                    "model_file": str(LETNET_MODEL_FILE),
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
    parser = argparse.ArgumentParser(description="Protocol-compliant NEU LETNet baseline training")
    parser.add_argument("--config", nargs="?", type=str, required=True, help="Configuration yaml")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    log_dir = _resolve(cfg["experiment"]["log_dir"])
    if "model_savePath" in str(log_dir):
        raise RuntimeError("Refusing historical directory: {}".format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_neu_letnet_240k")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_dir / "train.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    logger.info("Let the games begin (protocol baseline: LETNet NEU)")
    logger.info("Config: {}".format(args.config))

    train(cfg, logger)

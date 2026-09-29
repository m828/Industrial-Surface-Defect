# Protocol-compliant NEU-Seg baseline training: STDC-Seg (BiSeNet + STDCNet1446).
# Protocol (mirrors repro_runs/long240k_seed1337/base_b/code/train_long240k_base_b.py discipline):
#   * from-scratch, seed 1337, Adam lr=1e-4 (constant, no LR schedule), weight_decay=2e-6
#   * batch 16, input 200x200, 4 classes, train_iters from config (formal: 240000)
#   * NO val/test loader is ever constructed; NO model selection / best checkpoint
#   * checkpoint saved ONLY at the final iteration into an independent checkpoint_dir
#   * checkpoint dict carries "model_state" and "iter" keys (unified evaluator contract)
# Model: new (copy)/stdc.py  BiSeNet('STDCNet1446', 4, use_boundary_8=True)
#   train/eval forward returns (feat_out, feat_out16, feat_out32, feat_out_sp8);
#   feat_out/16/32 are 4-ch seg logits upsampled to input size, feat_out_sp8 is a
#   1-ch boundary logit map at 1/8 resolution.
# Loss (multi-output CE + boundary BCE, BiSeNet convention):
#   CE on outputs[0..2] with equal weight 1.0; BCE-with-logits on the 1-ch boundary
#   output vs the binary foreground mask (label1) downsampled to the boundary map
#   size, weight 1.0. Weights come from cfg["training"]["loss_weights"].

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
NEW_DIR = WORKSPACE / "new"
MODEL_FILE = NEW_COPY_DIR / "stdc.py"  # pinned model definition (registered alias "stdc")

for _p in (str(NEW_COPY_DIR),):  # tools/, datagenerator_neu, dataAug_new, stdcnet
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.utils import get_logger  # noqa: E402
from tools.optimizers import get_optimizer  # noqa: E402
from tools.schedulers import get_scheduler  # noqa: E402
from datagenerator_neu import DataGenerator  # noqa: E402
from dataAug_new import Compose, Transforms_PIL, ToTensor  # noqa: E402


def load_model_module():
    spec = importlib.util.spec_from_file_location("stdc_seg_model", str(MODEL_FILE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(n_classes, device):
    module = load_model_module()
    model = module.BiSeNet(
        backbone="STDCNet1446",
        n_classes=n_classes,
        pretrain_model="",
        use_boundary_8=True,
    ).to(device)
    return model


def compute_loss(outputs, labels, label1, cfg):
    """outputs: (out, out16, out32, out_sp8); CE x3 + boundary BCE x1."""
    weights = cfg["training"].get("loss_weights", [1.0, 1.0, 1.0, 1.0])
    seg_outs = outputs[:3]
    boundary_out = outputs[3]

    comps = []
    loss = 0.0
    for out, w in zip(seg_outs, weights[:3]):
        ce = F.cross_entropy(out, labels, ignore_index=255)
        comps.append(ce)
        loss = loss + ce * w

    # boundary supervision: binary foreground mask downsampled to boundary map size
    btarget = F.interpolate(
        label1.unsqueeze(1).float(), size=boundary_out.shape[-2:], mode="nearest"
    )
    bce = F.binary_cross_entropy_with_logits(boundary_out, btarget)
    comps.append(bce)
    loss = loss + bce * weights[3]
    return loss, comps


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
    run_id = experiment_cfg.get("run_id") or experiment_cfg.get("run_prefix", "neu_stdc_bisenet_240k")
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
    logger.info("Model: BiSeNet(backbone='STDCNet1446', n_classes={}, use_boundary_8=True) from {}".format(
        n_classes, MODEL_FILE))

    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"].get("lr_schedule"))
    logger.info("Using scheduler {} (protocol expects constant LR)".format(type(scheduler).__name__))

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
            label1 = label1.squeeze(1).to(device, dtype=torch.int64)

            optimizer.zero_grad()
            outputs = model(images)
            loss, comps = compute_loss(outputs, labels, label1, cfg)
            loss.backward()
            optimizer.step()
            scheduler.step()

            time_meter.append(time.time() - iter_t0)

            if step % int(cfg["training"]["print_interval"]) == 0 or step == train_iters:
                lr_now = optimizer.param_groups[0]["lr"]
                avg_t = sum(time_meter) / max(1, len(time_meter))
                time_meter = []
                msg = "Iter [{:d}/{:d}]  Loss: {:.4f}  comps: {}  LR: {:.2e}  Time/Iter: {:.4f}".format(
                    step, train_iters, loss.item(),
                    [round(c.item(), 6) for c in comps], lr_now, avg_t)
                print(msg)
                logger.info(msg)
                with open(metrics_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps({
                        "iter": step, "loss": loss.item(),
                        "comps": [c.item() for c in comps],
                        "lr": lr_now, "time_per_iter": avg_t,
                        "wall_time": time.time() - t_start,
                    }) + "\n")

            if step == train_iters:
                state = {
                    "iter": step,
                    "epoch": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "run_id": run_id,
                    "note": "final-iteration checkpoint (protocol); NOT test-selected",
                }
                save_path = _assert_safe_checkpoint_path(
                    checkpoint_dir / "{}_iter{:06d}.pkl".format(run_id, step))
                torch.save(state, str(save_path))
                logger.info("Saved final checkpoint: {}".format(save_path))

                summary = {
                    "run_id": run_id,
                    "model": "STDC-Seg (BiSeNet + STDCNet1446)",
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
    parser = argparse.ArgumentParser(description="Protocol-compliant NEU STDC-Seg baseline training")
    parser.add_argument("--config", nargs="?", type=str, required=True, help="Configuration yaml")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    log_dir = _resolve(cfg["experiment"]["log_dir"])
    if "model_savePath" in str(log_dir):
        raise RuntimeError("Refusing historical directory: {}".format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(str(log_dir))
    logger.info("Let the games begin (protocol baseline: STDC-Seg NEU)")
    logger.info("Config: {}".format(args.config))

    train(cfg, logger)

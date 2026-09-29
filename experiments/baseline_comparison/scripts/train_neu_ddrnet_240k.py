# Protocol-compliant NEU-Seg baseline training: DDRNet (DualResNet_imagenet).
# Base: "new/train_neu_ddr.py". Protocolized:
#   * ALL val/test loader, val-metric and best-checkpoint logic REMOVED
#   * constant LR (ConstantLR), Adam lr=1e-4, wd=2e-6, seed 1337, from scratch
#   * batch 16, input 200x200, 4 classes
#   * checkpoint saved ONLY at the final iteration into an independent checkpoint_dir
#   * checkpoint dict carries "model_state" and "iter" keys (unified evaluator contract)
# Model: new/model_ddr.py  DualResNet_imagenet(num_classes=4)
#   (DualResNet(BasicBlock, [2,2,2,2], planes=64, spp_planes=128, head_planes=128,
#    augment=True)); train forward returns [x_, x_extra] at input resolution;
#   eval returns a single tensor.
# Loss (mirrors the chain exactly): [ohem_cross_entropy, bce_with_logits_loss] on
# the two outputs with weights [1.0, 0.4] from cfg.

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
import yaml
from torch.utils import data

WORKSPACE = Path("/workspace/Industrial Surface Defect")
NEW_COPY_DIR = WORKSPACE / "new (copy)"
NEW_DIR = WORKSPACE / "new"
MODEL_FILE = NEW_DIR / "model_ddr.py"  # pinned model definition (registered alias "ddr")

for _p in (str(NEW_COPY_DIR),):  # tools/, datagenerator_neu, dataAug_new
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.loss import get_loss_function_new  # noqa: E402
from tools.utils import get_logger  # noqa: E402
from tools.optimizers import get_optimizer  # noqa: E402
from tools.schedulers import get_scheduler  # noqa: E402
from datagenerator_neu import DataGenerator  # noqa: E402
from dataAug_new import Compose, Transforms_PIL, ToTensor  # noqa: E402


def load_model_module():
    spec = importlib.util.spec_from_file_location("ddrnet_model", str(MODEL_FILE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(n_classes, device):
    module = load_model_module()
    model = module.DualResNet_imagenet(num_classes=n_classes).to(device)
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
    run_id = experiment_cfg.get("run_id") or experiment_cfg.get("run_prefix", "neu_ddrnet_240k")
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
    logger.info("Model: DualResNet_imagenet(num_classes={}) from {}".format(n_classes, MODEL_FILE))

    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"].get("lr_schedule"))
    logger.info("Using scheduler {} (protocol expects constant LR)".format(type(scheduler).__name__))

    loss_fn = get_loss_function_new(cfg)
    logger.info("Using loss {}".format(loss_fn))
    weights = cfg["training"].get("loss_weights", [1.0, 0.4])

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
            outputs = model(images)  # [x_, x_extra] already at input resolution

            losses = []
            for j, output, weight in zip(range(len(outputs)), outputs, weights):
                losses.append(loss_fn[j](output, labels) * weight)
            loss = sum(losses)

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
                    "model": "DDRNet (DualResNet_imagenet)",
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
    parser = argparse.ArgumentParser(description="Protocol-compliant NEU DDRNet baseline training")
    parser.add_argument("--config", nargs="?", type=str, required=True, help="Configuration yaml")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    log_dir = _resolve(cfg["experiment"]["log_dir"])
    if "model_savePath" in str(log_dir):
        raise RuntimeError("Refusing historical directory: {}".format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(str(log_dir))
    logger.info("Let the games begin (protocol baseline: DDRNet NEU)")
    logger.info("Config: {}".format(args.config))

    train(cfg, logger)

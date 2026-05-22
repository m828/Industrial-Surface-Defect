#!/usr/bin/env python3
"""Evaluate PIDNet-S on NEU-Seg test set: mIoU, per-class IoU, FPS, Params, FLOPs."""

import sys
import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Add the current working directory to path for imports (run from 'new (copy)')
sys.path.insert(0, os.getcwd())

from tools.computemIou import runningScore
from dataAug_new import Compose, Transforms_PIL, ToTensor, TestRescale


NEU_CLASSES = ['background', 'inclusion', 'patch', 'scratch']


def load_pidnet_s(weight_path, num_classes=4, device='cuda'):
    from pid import PIDNet
    model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32,
                   ppm_planes=96, head_planes=128, augment=False)
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    # Remove augment head keys if present (augment=False model won't have them)
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered)
    model.load_state_dict(model_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def evaluate_class_iou(model, test_txt, dataset_root, input_hw=(200, 200), device='cuda'):
    """Evaluate per-class IoU on NEU-Seg test set."""
    val_transforms = Compose([TestRescale(input_hw=input_hw), ToTensor()])

    # Read test list
    masks_paths = []
    imgs_paths = []
    with open(test_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                masks_paths.append(os.path.join(dataset_root, parts[0]))
                imgs_paths.append(os.path.join(dataset_root, parts[1]))

    from PIL import Image
    import copy

    n_classes = len(NEU_CLASSES)
    running_metrics = runningScore(n_classes)

    for idx in range(len(imgs_paths)):
        # Load image
        img = Image.open(imgs_paths[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Load mask
        mask_img = Image.open(masks_paths[idx])
        if mask_img.mode != 'L':
            mask_img = mask_img.convert('L')
        mask = np.array(mask_img)
        # Keep raw mask values for multi-class (don't binarize)
        mask_tensor = torch.from_numpy(mask).long()

        # Apply transform
        dummy_binary = copy.deepcopy(mask_img)
        dummy_binary = np.array(dummy_binary)
        dummy_binary[dummy_binary != 0] = 1
        dummy_binary = Image.fromarray(dummy_binary.astype(np.uint8))

        img_t, mask_t, _ = val_transforms(img, mask_img, dummy_binary)
        img_t = img_t.unsqueeze(0).to(device, dtype=torch.float)

        with torch.no_grad():
            output = model(img_t)
            pred = output.data.max(1)[1].cpu().numpy().squeeze()

        # Resize pred to original mask size if needed
        if pred.shape != mask.shape:
            from PIL import Image as PILImage
            pred_pil = PILImage.fromarray(pred.astype(np.uint8))
            pred_pil = pred_pil.resize((mask.shape[1], mask.shape[0]), PILImage.NEAREST)
            pred = np.array(pred_pil)

        running_metrics.update([mask], [pred])

    score, class_iou = running_metrics.get_scores()
    return score, class_iou


def measure_fps(model, input_hw=(200, 200), device='cuda', warmup=50, iterations=200):
    """Measure FPS."""
    model.eval()
    input_tensor = torch.randn(1, 3, *input_hw).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            model(input_tensor)
        torch.cuda.synchronize()
        elapsed = time.time() - start
    return iterations / elapsed


def measure_complexity(num_classes=4, input_hw=(200, 200)):
    """Measure Params, FLOPs, model size."""
    from pid import PIDNet
    model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32,
                   ppm_planes=96, head_planes=128, augment=False)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 * 1024)
        os.unlink(f.name)

    try:
        from thop import profile
        input_cpu = torch.randn(1, 3, *input_hw)
        flops, _ = profile(model, inputs=(input_cpu,), verbose=False)
        flops_g = flops / 1e9
    except Exception:
        flops_g = None

    return params, flops_g, size_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str,
                        default='model_savePaths/pid_neu_0119_neu_pid.pkl')
    parser.add_argument('--test_txt', type=str,
                        default='dataset/test_neu.txt')
    parser.add_argument('--dataset_root', type=str, default='dataset/')
    parser.add_argument('--input_hw', type=int, nargs=2, default=[200, 200])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Complexity stats
    print('=== Complexity Stats ===')
    params, flops_g, size_mb = measure_complexity(num_classes=4, input_hw=tuple(args.input_hw))
    print(f'Params: {params:.2f}M')
    if flops_g is not None:
        print(f'FLOPs: {flops_g:.2f}G')
    print(f'Model Size: {size_mb:.2f}MB')

    # 2. Load model
    print('\n=== Loading Model ===')
    model = load_pidnet_s(args.weight, num_classes=4, device=device)
    print(f'Weight: {args.weight}')

    # 3. FPS
    print('\n=== FPS ===')
    fps_200 = measure_fps(model, input_hw=(200, 200), device=device)
    print(f'FPS (200x200): {fps_200:.1f}')

    # 4. Per-class IoU
    print('\n=== Per-class IoU on NEU-Seg Test Set ===')
    score, class_iou = evaluate_class_iou(
        model, args.test_txt, args.dataset_root,
        input_hw=tuple(args.input_hw), device=device
    )

    oa = score["Overall Acc: \t"]
    ma = score["Mean Acc : \t"]
    miou = score["Mean IoU : \t"]
    fwa = score["FreqW Acc : \t"]
    print(f'Overall Accuracy: {oa:.4f}')
    print(f'Mean Accuracy: {ma:.4f}')
    print(f'Mean IoU: {miou:.4f}')
    print(f'FreqW Accuracy: {fwa:.4f}')
    print('\nPer-class IoU:')
    for k, v in class_iou.items():
        name = NEU_CLASSES[int(k)] if int(k) < len(NEU_CLASSES) else f'class_{k}'
        print(f'  {name} (class {k}): {v:.4f}')

    # 5. Save results
    results_path = 'experiments/results/pidnet_neu_results.txt'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        f.write('=== PIDNet-S on NEU-Seg ===\n')
        f.write(f'Weight: {args.weight}\n')
        f.write(f'Params: {params:.2f}M\n')
        if flops_g is not None:
            f.write(f'FLOPs: {flops_g:.2f}G\n')
        f.write(f'Model Size: {size_mb:.2f}MB\n')
        f.write(f'FPS (200x200): {fps_200:.1f}\n')
        f.write(f'Overall Accuracy: {oa:.4f}\n')

        f.write(f'Mean Accuracy: {ma:.4f}\n')
        f.write(f'Mean IoU: {miou:.4f}\n')
        f.write(f'FreqW Accuracy: {fwa:.4f}\n')
        f.write('\nPer-class IoU:\n')
        for k, v in class_iou.items():
            name = NEU_CLASSES[int(k)] if int(k) < len(NEU_CLASSES) else f'class_{k}'
            f.write(f'  {name} (class {k}): {v:.4f}\n')
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()

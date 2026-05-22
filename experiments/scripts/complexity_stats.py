#!/usr/bin/env python3
"""
复杂度统计脚本：计算 A2MS-DefectNet-S/B 和 Base-S/Base-B 的 Params、FLOPs、模型大小和 FPS。

使用方法：
    cd /workspace/Industrial\ Surface\ Defect/new
    /opt/conda/bin/python /workspace/Industrial\ Surface\ Defect/Industrial-Surface-Defect/experiments/scripts/complexity_stats.py

输出：
    - 终端表格
    - CSV 文件：experiments/results/complexity_results.csv
"""

import sys
import os
import csv
import time
import torch
import torch.nn as nn

# 添加 new/ 目录到 Python 路径
NEW_DIR = "/workspace/Industrial Surface Defect/new"
sys.path.insert(0, NEW_DIR)

from model_resnet import resnet18, resnet50


def count_parameters(model):
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """统计全部参数量"""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model):
    """计算模型大小 (MB)"""
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024


def measure_fps(model, input_size=(1, 3, 200, 200), device='cuda', warmup=50, runs=200):
    """测量 FPS"""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size).to(device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            model(dummy_input)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return runs / elapsed


def measure_fps_leather(model, input_size=(1, 3, 768, 768), device='cuda', warmup=20, runs=100):
    """测量皮革数据集输入尺寸的 FPS"""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size).to(device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            model(dummy_input)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return runs / elapsed


def compute_flops(model, input_size=(1, 3, 200, 200)):
    """使用 thop 计算 FLOPs"""
    try:
        from thop import profile
        model.eval()
        dummy_input = torch.randn(*input_size)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9  # 转换为 GFLOPs
    except Exception as e:
        print(f"  [警告] thop 计算失败: {e}")
        return None


def load_model_dsmonet(model_file, num_classes, backbone_fn, **kwargs):
    """从指定文件加载 DSMONet 模型"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.DSMONet(num_classes=num_classes, backbone=backbone_fn(), **kwargs)
    return model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print()

    # 模型配置
    # NEU-Seg: 200x200, 4 classes (background + 3 defect types)
    # Leather: 768x768, 8 classes (background + 7 defect types)
    configs = [
        {
            'name': 'A2MS-DefectNet-S',
            'model_file': os.path.join(NEW_DIR, 'model_dsmo_512_eSE.py'),
            'backbone_fn': resnet18,
            'num_classes_neu': 4,
            'num_classes_leather': 8,
            'backbone': 'ResNet-18',
        },
        {
            'name': 'A2MS-DefectNet-B',
            'model_file': os.path.join(NEW_DIR, 'model_dsmo_rs50_eSE_adapt_detailloss.py'),
            'backbone_fn': resnet50,
            'num_classes_neu': 4,
            'num_classes_leather': 8,
            'backbone': 'ResNet-50',
        },
        {
            'name': 'Base-S',
            'model_file': os.path.join(NEW_DIR, 'model_dsmo_512.py'),
            'backbone_fn': resnet18,
            'num_classes_neu': 4,
            'num_classes_leather': 8,
            'backbone': 'ResNet-18',
        },
        {
            'name': 'Base-B',
            'model_file': os.path.join(NEW_DIR, 'model_dsmo_rs50.py'),
            'backbone_fn': resnet50,
            'num_classes_neu': 4,
            'num_classes_leather': 8,
            'backbone': 'ResNet-50',
        },
    ]

    results = []

    for cfg in configs:
        print(f"{'='*60}")
        print(f"模型: {cfg['name']} ({cfg['backbone']})")
        print(f"{'='*60}")

        # 加载模型（使用 NEU-Seg 的类别数）
        try:
            model = load_model_dsmonet(
                cfg['model_file'],
                num_classes=cfg['num_classes_neu'],
                backbone_fn=cfg['backbone_fn']
            )
        except Exception as e:
            print(f"  [错误] 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'model': cfg['name'],
                'backbone': cfg['backbone'],
                'params_m': 'ERROR',
                'flops_g_200': 'ERROR',
                'model_size_mb': 'ERROR',
                'fps_200': 'ERROR',
                'fps_768': 'ERROR',
            })
            continue

        # 参数量
        params = count_all_parameters(model)
        params_m = params / 1e6
        print(f"  Params: {params:,} ({params_m:.2f}M)")

        # 模型大小
        model_size = get_model_size_mb(model)
        print(f"  Model Size: {model_size:.2f} MB")

        # FLOPs (200x200)
        flops_200 = compute_flops(model, input_size=(1, 3, 200, 200))
        if flops_200 is not None:
            print(f"  FLOPs (200x200): {flops_200:.2f}G")

        # FPS (200x200, NEU-Seg)
        print(f"  测量 FPS (200x200)...")
        fps_200 = measure_fps(model, input_size=(1, 3, 200, 200), device=device)
        print(f"  FPS (200x200): {fps_200:.1f}")

        # FPS (768x768, Leather)
        print(f"  测量 FPS (768x768)...")
        fps_768 = measure_fps_leather(model, input_size=(1, 3, 768, 768), device=device)
        print(f"  FPS (768x768): {fps_768:.1f}")

        print()

        results.append({
            'model': cfg['name'],
            'backbone': cfg['backbone'],
            'params_m': f"{params_m:.2f}",
            'flops_g_200': f"{flops_200:.2f}" if flops_200 else 'N/A',
            'model_size_mb': f"{model_size:.2f}",
            'fps_200': f"{fps_200:.1f}",
            'fps_768': f"{fps_768:.1f}",
        })

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # 打印汇总表
    print("\n" + "="*80)
    print("汇总表")
    print("="*80)
    print(f"{'模型':<20} {'Backbone':<12} {'Params(M)':<12} {'FLOPs(G)':<12} {'Size(MB)':<12} {'FPS@200':<10} {'FPS@768':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['model']:<20} {r['backbone']:<12} {r['params_m']:<12} {r['flops_g_200']:<12} {r['model_size_mb']:<12} {r['fps_200']:<10} {r['fps_768']:<10}")

    # 保存 CSV
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'complexity_results.csv'
    )
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model', 'dataset', 'input_size', 'backbone', 'params_m',
            'flops_g', 'model_size_mb', 'fps', 'batch_size', 'device',
            'script', 'weight_path', 'date', 'verified'
        ])
        writer.writeheader()
        for r in results:
            # 200x200 (NEU-Seg)
            writer.writerow({
                'model': r['model'],
                'dataset': 'NEU-Seg',
                'input_size': '200x200',
                'backbone': r['backbone'],
                'params_m': r['params_m'],
                'flops_g': r['flops_g_200'],
                'model_size_mb': r['model_size_mb'],
                'fps': r['fps_200'],
                'batch_size': 1,
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'script': 'experiments/scripts/complexity_stats.py',
                'weight_path': 'N/A (random init)',
                'date': time.strftime('%Y-%m-%d'),
                'verified': 'no',
            })
            # 768x768 (Leather)
            writer.writerow({
                'model': r['model'],
                'dataset': 'Leather',
                'input_size': '768x768',
                'backbone': r['backbone'],
                'params_m': r['params_m'],
                'flops_g': 'N/A',
                'model_size_mb': r['model_size_mb'],
                'fps': r['fps_768'],
                'batch_size': 1,
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'script': 'experiments/scripts/complexity_stats.py',
                'weight_path': 'N/A (random init)',
                'date': time.strftime('%Y-%m-%d'),
                'verified': 'no',
            })

    print(f"\n结果已保存到: {csv_path}")

    # 同时输出待核验格式
    pending_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'new_results_pending.md'
    )
    print(f"\n请将以下内容追加到 {pending_path}:")


if __name__ == '__main__':
    main()

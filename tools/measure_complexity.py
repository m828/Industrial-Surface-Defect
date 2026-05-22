#!/usr/bin/env python3
"""
复杂度统计脚本：计算所有论文模型的 Params、FLOPs、模型大小和 FPS。

目标模型列表：
    1.  Base-S
    2.  Base-B
    3.  A2MS-DefectNet-S
    4.  A2MS-DefectNet-B
    5.  DDRNet23slim
    6.  STDC1-Seg
    7.  STDC2-Seg
    8.  PP-LiteSeg-T
    9.  PP-LiteSeg-B
    10. BiSeNetV1-L
    11. BiSeNetV2-L (未找到代码文件)
    12. Sub-region UNet
    13. PIDNet-S

使用方法：
    cd /workspace/Industrial\\ Surface\\ Defect/new
    /opt/conda/bin/python /workspace/Industrial\\ Surface\\ Defect/Industrial-Surface-Defect/tools/measure_complexity.py

输出：
    - 终端表格
    - CSV 文件：experiments/results/complexity_results.csv
    - 汇总文件：experiments/results/complexity_summary.md

约束：
    - 不修改任何模型结构
    - 不编造数据
    - 模型导入失败或 forward 失败时记录错误
    - 所有结果可复现
"""

import sys
import os
import csv
import time
import json
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 路径配置
# ============================================================
WORKSPACE = "/workspace/Industrial Surface Defect"
NEW_DIR = os.path.join(WORKSPACE, "new")
NEW_COPY_DIR = os.path.join(WORKSPACE, "new (copy)")
SUBREGION_DIR = os.path.join(WORKSPACE, "subregion unet")

# 将 new/ 加入 sys.path（大量模型依赖 tools/ 和 stdcnet 等模块）
sys.path.insert(0, NEW_DIR)
# 同时加入 new (copy)/ 以支持 STDC BiSeNet 等
sys.path.insert(0, NEW_COPY_DIR)

# ============================================================
# 工具函数
# ============================================================

def count_all_parameters(model):
    """统计全部参数量（含不可训练）"""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model):
    """计算模型大小 (MB)，含参数和缓冲区"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def compute_flops(model, input_size=(1, 3, 200, 200)):
    """使用 thop 计算 FLOPs（单位：G），确保模型在 CPU 上，完成后清理 hooks"""
    try:
        from thop import profile
        model.eval()
        model.cpu()
        dummy_input = torch.randn(*input_size)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        # 清理 thop 注册的 hooks，避免后续 forward 时访问 total_ops 失败
        _remove_thop_hooks(model)
        return flops / 1e9
    except Exception as e:
        return f"ERROR: {e}"


def _remove_thop_hooks(model):
    """移除 thop 注册的 forward hooks，并清理 total_ops/total_params 属性"""
    # 清理属性
    for m in model.modules():
        if hasattr(m, 'total_ops'):
            try:
                del m.total_ops
            except AttributeError:
                pass
        if hasattr(m, 'total_params'):
            try:
                del m.total_params
            except AttributeError:
                pass
    # 移除 hooks（thop 使用 register_forward_hook）
    # PyTorch 2.x 提供 _get_hooks 方法
    for m in model.modules():
        if hasattr(m, '_forward_hooks'):
            m._forward_hooks.clear()
        if hasattr(m, '_forward_pre_hooks'):
            # 不清除 pre-hooks，可能有用户注册的
            pass


def measure_fps(model, input_size=(1, 3, 200, 200), device='cuda', warmup=50, runs=200):
    """测量 FPS（含 cuda.synchronize）"""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size).to(device)

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


def test_forward(model, input_size=(1, 3, 200, 200), device='cpu'):
    """测试模型 forward 是否能正常运行（默认 CPU，避免设备不一致）"""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    return output


def load_module_from_file(filepath):
    """动态加载 Python 模块"""
    spec = importlib.util.spec_from_file_location("model_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ListOutputWrapper(nn.Module):
    """包装返回 list 的模型，使 thop 能正确计算 FLOPs"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


# ============================================================
# 模型工厂：每个模型的实例化逻辑
# ============================================================

def create_base_s(num_classes=4):
    """Base-S: ResNet-18 + DSMONet (SELayer, 无 eSE, 无 detailloss)"""
    from model_resnet import resnet18
    module = load_module_from_file(os.path.join(NEW_COPY_DIR, "model_dsmo_rs18.py"))
    model = module.DSMONet(
        num_classes=num_classes,
        backbone=resnet18(),
        backbone_indices=[1, 2, 3, 4],
        out_ch=128
    )
    return model, "ResNet-18"


def create_base_b(num_classes=4):
    """Base-B: ResNet-50 + DSMONet (SELayer, 无 eSE, 无 detailloss)"""
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50.py"))
    model = module.DSMONet(
        num_classes=num_classes,
        backbone=resnet50(),
        backbone_indices=[0, 1, 2, 3, 4],
        out_ch=128
    )
    return model, "ResNet-50"


def create_a2ms_defectnet_s(num_classes=4):
    """A2MS-DefectNet-S: ResNet-18 + eSE + AdaptiveChannelWeight + detailloss"""
    from model_resnet import resnet18
    module = load_module_from_file(os.path.join(NEW_COPY_DIR, "model_dsmo_rs18_eSE_adapt_detailloss_822.py"))
    model = module.DSMONet(
        num_classes=num_classes,
        backbone=resnet18(),
        backbone_indices=[1, 2, 3, 4],
        out_ch=128
    )
    return model, "ResNet-18"


def create_a2ms_defectnet_b(num_classes=4):
    """A2MS-DefectNet-B: ResNet-50 + eSE + AdaptiveChannelWeight + detailloss"""
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50_eSE_adapt_detailloss.py"))
    model = module.DSMONet(
        num_classes=num_classes,
        backbone=resnet50(),
        backbone_indices=[0, 1, 2, 3, 4],
        out_ch=128
    )
    return model, "ResNet-50"


def create_ddrnet23slim(num_classes=4):
    """DDRNet23slim: 内置 backbone，DSMONet(block=BasicBlock, layers=[2,2,2,2], planes=32)"""
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_ddr_s.py"))
    model = module.DSMONet(
        block=module.BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        planes=32,
        spp_planes=128,
        head_planes=64,
        augment=False
    )
    return model, "DDRNet-Internal"


def create_stdc1_seg(num_classes=4):
    """STDC1-Seg: STDCNet1446 backbone + BiSeNet 语义分割头"""
    # 需要从 new (copy)/stdc.py 加载 BiSeNet
    module = load_module_from_file(os.path.join(NEW_COPY_DIR, "stdc.py"))
    model = module.BiSeNet(
        backbone='STDCNet1446',
        n_classes=num_classes,
        pretrain_model='',
        use_boundary_8=True
    )
    return model, "STDCNet1446"


def create_stdc2_seg(num_classes=4):
    """STDC2-Seg: STDCNet813 backbone + BiSeNet 语义分割头"""
    module = load_module_from_file(os.path.join(NEW_COPY_DIR, "stdc.py"))
    model = module.BiSeNet(
        backbone='STDCNet813',
        n_classes=num_classes,
        pretrain_model='',
        use_boundary_8=True
    )
    return model, "STDCNet813"


def create_ppliteseg_t(num_classes=4):
    """PP-LiteSeg-T: 轻量版 PP-LiteSeg

    注意：model_dsmo_rs50_ppliteseg.py 中 backbone_out_chs 硬编码为 [512,1024,2048]，
    仅兼容 ResNet-50。ResNet-18 的输出通道为 [128,256,512]，与 SPPM 不匹配。
    因此 PP-LiteSeg-T 使用 ResNet-50 backbone，与 PP-LiteSeg-B 共用同一文件。
    T/B 区分可能在其他配置中（如 out_ch 或训练策略），此处标记为需要人工确认。
    """
    raise NotImplementedError(
        "PP-LiteSeg-T: model_dsmo_rs50_ppliteseg.py 的 backbone_out_chs 硬编码为 [512,1024,2048]，"
        "仅支持 ResNet-50。PP-LiteSeg-T (ResNet-18) 需要单独的模型文件或修改 backbone_out_chs，"
        "但约束不允许修改模型结构。需人工确认 PP-LiteSeg-T 的正确代码入口。"
    )


def create_ppliteseg_b(num_classes=4):
    """PP-LiteSeg-B: 标准版 PP-LiteSeg (ResNet-50 backbone, backbone_indices=[0,1,2,3,4])"""
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50_ppliteseg.py"))
    model = module.DSMONet(
        num_classes=num_classes,
        backbone=resnet50(),
        backbone_indices=[0, 1, 2, 3, 4],
        out_ch=128
    )
    return model, "ResNet-50"


def create_bisenetv1_l(num_classes=4):
    """BiSeNetV1-L: DSMONet from csfcn_yuan.py (含 CFC_CRB + SFC_G2 + PSPModule + LocalAttenModule)"""
    from model_resnet import resnet50
    module = load_module_from_file(os.path.join(NEW_DIR, "model_dsmo_rs50_csfcn_yuan.py"))
    # CSFCN 类的 Backbone() 未定义，使用同文件的 DSMONet 类（含 CFC_CRB + SFC_G2）
    model = module.DSMONet(
        num_classes=num_classes,
        backbone=resnet50(),
        backbone_indices=[0, 1, 2, 3, 4],
        out_ch=128
    )
    return model, "ResNet-50"


def create_subregion_unet(num_classes=4):
    """Sub-region UNet: fianlModel (Sub-region 分区 UNet)

    注意：fianlModel 的 inchannel 默认为 16，forward 期望输入为 16 通道（经 pixelshuffle_invert 处理后）。
    直接输入 3 通道 RGB 图像会导致通道不匹配。
    需要人工确认：是否需要 pixelshuffle_invert 预处理，或 inchannel 应设为 3。
    """
    module = load_module_from_file(os.path.join(SUBREGION_DIR, "model_p.py"))
    # 尝试 inchannel=3 以匹配 RGB 输入
    model = module.fianlModel(inchannel=3, nclass=num_classes)
    return model, "Sub-region-UNet"


def create_pidnet_s(num_classes=4):
    """PIDNet-S: PIDNet(m=2, n=3, planes=32, ppm_planes=96, head_planes=128)"""
    module = load_module_from_file(os.path.join(NEW_DIR, "pid.py"))
    model = module.PIDNet(
        m=2,
        n=3,
        num_classes=num_classes,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=True
    )
    return model, "PIDNet-Internal"


# ============================================================
# 模型配置表
# ============================================================

MODEL_CONFIGS = [
    {
        'name': 'Base-S',
        'factory': create_base_s,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
    {
        'name': 'Base-B',
        'factory': create_base_b,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
    {
        'name': 'A2MS-DefectNet-S',
        'factory': create_a2ms_defectnet_s,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
    {
        'name': 'A2MS-DefectNet-B',
        'factory': create_a2ms_defectnet_b,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
    {
        'name': 'DDRNet23slim',
        'factory': create_ddrnet23slim,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
    {
        'name': 'STDC1-Seg',
        'factory': create_stdc1_seg,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
    {
        'name': 'STDC2-Seg',
        'factory': create_stdc2_seg,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
    {
        'name': 'PP-LiteSeg-T',
        'factory': create_ppliteseg_t,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
    {
        'name': 'PP-LiteSeg-B',
        'factory': create_ppliteseg_b,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
    {
        'name': 'BiSeNetV1-L',
        'factory': create_bisenetv1_l,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
    {
        'name': 'BiSeNetV2-L',
        'factory': None,
        'input_sizes': {},
        'skip': True,
        'skip_reason': '服务器上未找到 BiSeNetV2 代码文件，需从外部引入',
    },
    {
        'name': 'Sub-region UNet',
        'factory': create_subregion_unet,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
    {
        'name': 'PIDNet-S',
        'factory': create_pidnet_s,
        'input_sizes': {
            'NEU-Seg': (1, 3, 200, 200),
            'Leather': (1, 3, 768, 768),
        },
        'skip': False,
    },
]


# ============================================================
# 主流程
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"日期: {time.strftime('%Y-%m-%d')}")
    print()

    results = []
    errors = []

    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        print(f"{'='*60}")
        print(f"模型: {name}")
        print(f"{'='*60}")

        # 跳过不可用模型
        if cfg.get('skip'):
            reason = cfg.get('skip_reason', '未知原因')
            print(f"  [跳过] {reason}")
            errors.append({'model': name, 'error': reason})
            print()
            continue

        # 1. 实例化模型
        try:
            model, backbone_name = cfg['factory'](num_classes=4)
            print(f"  Backbone: {backbone_name}")
        except Exception as e:
            err_msg = f"实例化失败: {e}"
            print(f"  [错误] {err_msg}")
            import traceback
            traceback.print_exc()
            errors.append({'model': name, 'error': err_msg})
            print()
            continue

        # 2. 测试 forward (200x200) — 在 CPU 上测试，避免设备不一致
        try:
            output = test_forward(model, input_size=(1, 3, 200, 200), device='cpu')
            if isinstance(output, (list, tuple)):
                out_shape = [o.shape for o in output]
                print(f"  Forward 测试通过 (list of {len(output)}): {out_shape}")
            else:
                print(f"  Forward 测试通过: {output.shape}")
        except Exception as e:
            err_msg = f"Forward 失败 (200x200): {e}"
            print(f"  [错误] {err_msg}")
            import traceback
            traceback.print_exc()
            errors.append({'model': name, 'error': err_msg})
            del model
            torch.cuda.empty_cache()
            print()
            continue

        # 3. 参数量
        params = count_all_parameters(model)
        params_m = params / 1e6
        print(f"  Params: {params:,} ({params_m:.2f}M)")

        # 4. 模型大小
        model_size = get_model_size_mb(model)
        print(f"  Model Size: {model_size:.2f} MB")

        # 5. FPS (200x200) — 在 FLOPs 之前测量，避免 thop hooks 干扰
        try:
            print(f"  测量 FPS (200x200, warmup=50, runs=200)...")
            fps_200 = measure_fps(model, input_size=(1, 3, 200, 200), device=device)
            print(f"  FPS (200x200): {fps_200:.1f}")
        except Exception as e:
            fps_200 = None
            print(f"  [错误] FPS 测量失败 (200x200): {e}")

        # 6. FPS (768x768)
        try:
            print(f"  测量 FPS (768x768, warmup=50, runs=200)...")
            fps_768 = measure_fps(model, input_size=(1, 3, 768, 768), device=device)
            print(f"  FPS (768x768): {fps_768:.1f}")
        except Exception as e:
            fps_768 = None
            print(f"  [错误] FPS 测量失败 (768x768): {e}")

        # 7. FLOPs (200x200) — 放在最后，因为 thop 会注册 hooks
        wrapped = ListOutputWrapper(model)
        flops_200 = compute_flops(wrapped, input_size=(1, 3, 200, 200))
        if isinstance(flops_200, str):
            print(f"  FLOPs (200x200): {flops_200}")
        else:
            print(f"  FLOPs (200x200): {flops_200:.2f}G")

        print()

        # 记录结果
        for dataset_name, input_size in cfg['input_sizes'].items():
            h, w = input_size[2], input_size[3]
            fps_val = fps_200 if dataset_name == 'NEU-Seg' else fps_768
            results.append({
                'model': name,
                'dataset': dataset_name,
                'input_size': f'{h}x{w}',
                'backbone': backbone_name,
                'params_m': f"{params_m:.2f}",
                'flops_g': f"{flops_200:.2f}" if (dataset_name == 'NEU-Seg' and isinstance(flops_200, float)) else 'N/A',
                'model_size_mb': f"{model_size:.2f}",
                'fps': f"{fps_val:.1f}" if fps_val else 'ERROR',
                'batch_size': 1,
                'device': gpu_name,
                'script': 'tools/measure_complexity.py',
                'weight_path': 'N/A (random init)',
                'date': time.strftime('%Y-%m-%d'),
                'verified': 'no',
            })

        # 释放显存
        del model
        try:
            del wrapped
        except NameError:
            pass
        torch.cuda.empty_cache()

    # ============================================================
    # 输出汇总
    # ============================================================
    print("\n" + "="*100)
    print("汇总表 (Params / FLOPs / Size / FPS@200 / FPS@768)")
    print("="*100)
    header = f"{'模型':<20} {'Backbone':<16} {'Params(M)':<12} {'FLOPs(G)':<12} {'Size(MB)':<12} {'FPS@200':<10} {'FPS@768':<10}"
    print(header)
    print("-"*100)

    # 去重打印（每个模型只打印一行）
    printed = set()
    for r in results:
        if r['model'] not in printed:
            # 找对应的 768 FPS
            fps_768_row = [x for x in results if x['model'] == r['model'] and x['dataset'] == 'Leather']
            fps_768_val = fps_768_row[0]['fps'] if fps_768_row else 'N/A'
            print(f"{r['model']:<20} {r['backbone']:<16} {r['params_m']:<12} {r['flops_g']:<12} {r['model_size_mb']:<12} {r['fps']:<10} {fps_768_val:<10}")
            printed.add(r['model'])

    # 打印错误
    if errors:
        print("\n" + "="*100)
        print("错误/跳过记录")
        print("="*100)
        for e in errors:
            print(f"  {e['model']}: {e['error']}")

    # ============================================================
    # 保存 CSV
    # ============================================================
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'experiments', 'results'
    )
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, 'complexity_results.csv')
    fieldnames = [
        'model', 'dataset', 'input_size', 'backbone', 'params_m',
        'flops_g', 'model_size_mb', 'fps', 'batch_size', 'device',
        'script', 'weight_path', 'date', 'verified'
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nCSV 已保存到: {csv_path}")

    # ============================================================
    # 保存汇总 Markdown
    # ============================================================
    summary_path = os.path.join(results_dir, 'complexity_summary.md')
    with open(summary_path, 'w') as f:
        f.write("# 复杂度统计汇总\n\n")
        f.write(f"- 日期: {time.strftime('%Y-%m-%d')}\n")
        f.write(f"- 设备: {gpu_name}\n")
        f.write(f"- PyTorch: {torch.__version__}\n")
        f.write(f"- 统计脚本: `tools/measure_complexity.py`\n")
        f.write(f"- 权重: 随机初始化（未加载训练权重）\n")
        f.write(f"- FPS 测量: warmup=50, runs=200, batch_size=1, cuda.synchronize\n\n")

        f.write("## 结果表格\n\n")
        f.write("| 模型 | Backbone | Params(M) | FLOPs(G) 200x200 | Size(MB) | FPS@200 | FPS@768 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        printed = set()
        for r in results:
            if r['model'] not in printed:
                fps_768_row = [x for x in results if x['model'] == r['model'] and x['dataset'] == 'Leather']
                fps_768_val = fps_768_row[0]['fps'] if fps_768_row else 'N/A'
                f.write(f"| {r['model']} | {r['backbone']} | {r['params_m']} | {r['flops_g']} | {r['model_size_mb']} | {r['fps']} | {fps_768_val} |\n")
                printed.add(r['model'])

        if errors:
            f.write("\n## 错误/跳过记录\n\n")
            for e in errors:
                f.write(f"- **{e['model']}**: {e['error']}\n")

        f.write("\n## 说明\n\n")
        f.write("- FLOPs 仅统计 200x200 输入（NEU-Seg），768x768 输入的 FLOPs 未计算\n")
        f.write("- FPS 为推理速度（eval 模式，不含后处理）\n")
        f.write("- 模型大小包含参数和缓冲区\n")
        f.write("- 所有模型使用随机初始化权重\n")

    print(f"汇总已保存到: {summary_path}")

    # ============================================================
    # 保存错误 JSON（便于脚本解析）
    # ============================================================
    if errors:
        error_path = os.path.join(results_dir, 'complexity_errors.json')
        with open(error_path, 'w') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"错误记录已保存到: {error_path}")

    print("\n完成！")


if __name__ == '__main__':
    main()

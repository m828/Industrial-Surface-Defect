"""Inference Conv2d/ConvTranspose2d/Linear MAC ledger, not a latency benchmark."""
import argparse, json
from pathlib import Path
import torch
from ..models import build_model


def profile(model, shape=(1,3,200,200)):
    count = 0
    output_grid_count = 0
    handles=[]
    def hook(m, inputs, out):
        nonlocal count, output_grid_count
        if isinstance(m, torch.nn.ConvTranspose2d):
            count += inputs[0].numel()*(m.out_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
            output_grid_count += out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        elif isinstance(m, torch.nn.Conv2d):
            count += out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
            output_grid_count += out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        elif isinstance(m, torch.nn.Linear):
            count += out.numel()*m.in_features
            output_grid_count += out.numel()*m.in_features
    for m in model.modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.ConvTranspose2d,torch.nn.Linear)):
            handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad(): model(torch.randn(shape))
    for h in handles: h.remove()
    return {'parameters_total':sum(p.numel() for p in model.parameters()),
            'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'inference_input_lattice_macs':count, 'inference_output_grid_macs':output_grid_count,'shape':shape,
            'excludes':'pooling, normalization, activation, resize, grid_sample, elementwise ops and memory/latency'}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
    torch.set_num_threads(4)
    result={v:profile(build_model(v)) for v in ['formal_base','formal_a2ms','coupled_transport','capacity_control']}
    result['scope']='synthetic CPU operation ledger; not performance or historical-complexity replacement'
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

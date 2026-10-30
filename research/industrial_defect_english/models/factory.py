"""Read-only formal-source adapter and independent experimental model factory.

Only unavailable, unused legacy imports are removed with AST. Vendor bytes stay
unchanged. The backbone's download helper is injected from torch.hub; no weights
are downloaded (pretrained=False). No global sys.modules mocks are installed.
"""
from pathlib import Path
import ast
import hashlib
import json
import types
import torch

VENDOR = Path(__file__).resolve().parents[1] / 'vendor'


def _load(name, filename, removed):
    path = VENDOR / filename
    expected = next(x['sha256'] for x in json.loads((VENDOR/'provenance.json').read_text())
                    if Path(x['copy']).name == filename)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Vendor SHA mismatch: {filename}')
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not
                 (isinstance(node, ast.ImportFrom) and node.module in removed)]
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    mod.__dict__['load_state_dict_from_url'] = torch.hub.load_state_dict_from_url
    exec(compile(tree, str(path), 'exec'), mod.__dict__)
    return mod


FORMAL_BASE = _load('formal_base', 'model_dsmo_rs50.py', {'tools.models.utils', 'stdcnet'})
FORMAL_A2 = _load('formal_a2', 'model_dsmo_rs50_eSE_adapt_detailloss_fixed.py',
                  {'tools.models.utils', 'stdcnet'})
BACKBONE = _load('formal_backbone', 'model_resnet.py', {'utils'})


def build_model(variant='formal_a2ms', num_classes=4, seed=1337):
    from .transport import CoupledTransportModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = BACKBONE.resnet50(pretrained=False)
        if variant == 'formal_base':
            return FORMAL_BASE.DSMONet(num_classes, backbone)
        if variant == 'formal_a2ms':
            return FORMAL_A2.DSMONet(num_classes, backbone)
        if variant in ('coupled_transport', 'capacity_control', 'head_shift_only',
                       'uncoupled_transport', 'no_head_shift', 'stage1_only', 'recomputed_final_transport'):
            # All old parameter tensors have identical seeded initialization.
            return CoupledTransportModel(num_classes, backbone, variant=variant)
        raise ValueError(variant)


def load_compatible(model, state, allow_new=False):
    """state is a tensor-only state_dict, not an automatically unpickled file."""
    result = model.load_state_dict(state, strict=False)
    permitted = {k for k in model.state_dict() if k.startswith('transport.')}
    if result.unexpected_keys or any(k not in permitted for k in result.missing_keys):
        raise RuntimeError(f'Incompatible state: {result}')
    if result.missing_keys and not allow_new:
        raise RuntimeError('Explicit allow_new=True required for new parameters')
    return {'missing': result.missing_keys, 'unexpected': result.unexpected_keys}

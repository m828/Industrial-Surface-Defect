"""Read-only, SHA-required CPU tensor loading; no model forward and no unsafe pickle fallback."""
import argparse,hashlib,json
from pathlib import Path
import torch
from ..models import build_model,load_compatible


def verify(path,expected):
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual!=expected:raise ValueError('Checkpoint SHA mismatch')
    obj=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(obj,dict):raise ValueError('Expected mapping checkpoint')
    state=obj
    for key in ['model_state','state_dict','model']:
        if key in obj and isinstance(obj[key],dict):state=obj[key];break
    if not state or not all(isinstance(k,str) and torch.is_tensor(v) for k,v in state.items()):
        raise ValueError('No tensor-only state dict; no unsafe fallback allowed')
    prefixed=all(k.startswith('module.') for k in state)
    if prefixed:state={k[7:]:v for k,v in state.items()}
    formal=build_model('formal_a2ms');formal.load_state_dict(state,strict=True)
    result={'checkpoint':str(Path(path).resolve()),'sha256':actual,'stripped_module_prefix':prefixed,
            'formal_a2ms_strict':'PASS','scope':'State loading only, no forward. Historical weights do not initialize E0.'}
    for variant in ['coupled_transport','capacity_control']:
        result[variant]=load_compatible(build_model(variant),state,allow_new=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=verify(a.checkpoint,a.expected_sha)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2)
    print('CHECKPOINT_LOAD_PASS_NO_FORWARD')

if __name__=='__main__':main()

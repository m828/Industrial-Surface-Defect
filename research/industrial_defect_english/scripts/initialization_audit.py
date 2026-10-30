"""CPU construction/tensor hashing only. No forward, optimizer, checkpoint or data."""
import argparse,hashlib
from .handoff_common import MATRIX,write_new_json


def tensor_sha(tensor):
    value=tensor.detach().cpu().contiguous()
    return hashlib.sha256(str(value.dtype).encode()+str(tuple(value.shape)).encode()+value.numpy().tobytes()).hexdigest()


def audit(seed=1337):
    from ..models import build_model
    first=None;details={};errors=[]
    for variant in MATRIX:
        model=build_model(variant,4,seed)
        common={k:tensor_sha(v) for k,v in model.state_dict().items() if not k.startswith('transport.')}
        if first is None:first=common
        if common!=first:errors.append(variant+':common_initial_tensor_mismatch')
        details[variant]={'total_parameters':sum(p.numel() for p in model.parameters()),
                          'common_tensor_count':len(common),
                          'common_tensor_collection_sha256':hashlib.sha256(str(sorted(common.items())).encode()).hexdigest(),
                          'new_parameters':sum(p.numel() for n,p in model.named_parameters() if n.startswith('transport.'))}
        del model
    return {'status':'PASS' if not errors else 'FAIL','seed':seed,'checks':details,'errors':errors,
            'scope':'CPU construction only; no forward/optimizer; same common tensors, not performance',
            'cuda_rng_contract':'Reset seed after construction in approved runner; no stochastic GPU fairness claimed by CPU hashes'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=1337);a=p.parse_args()
    result=audit(a.seed);write_new_json(a.output,result);return 0 if result['status']=='PASS' else 2

if __name__=='__main__':raise SystemExit(main())

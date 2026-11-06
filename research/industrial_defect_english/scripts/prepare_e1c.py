"""Derive three E1-C configs from real completed coupled runs; no data freeze."""
import argparse,copy,hashlib,json,subprocess
from pathlib import Path
from .handoff_common import E1C_MATRIX,sha256,execution_protocol_errors
from .prepare_e1a import FROZEN_MODELS
from .prepare_e1b import read_reference,SIGNATURE_KEYS,E1A_RESULTS_COMMIT

BASE_COMMIT='faed7033e78e5a7a9e6dc5a2f798545aa31e52e7'
ROOT=Path(__file__).resolve().parents[1]
SHARED=ROOT/'configs/e1c_protocol_v1.json'
FROZEN_ASSETS=json.loads(SHARED.read_text())['frozen_asset_sha256']


def reference(root):
    root=Path(root).resolve()
    if not (root/'run.json').is_file():raise ValueError('E1-C requires the real reference run.json: '+str(root))
    record=read_reference(root)
    seed,cfg,meta,identity=record
    for key,value in FROZEN_ASSETS.items():
        if cfg.get(key)!=value:raise ValueError('Reference differs from frozen E0/E1-A identity: '+key)
    return record


def check_reference_binding(cfg):
    seed,base,meta,identity=reference(cfg.get('reference_coupled_run_directory',''))
    if seed!=cfg.get('seed') or identity!=cfg.get('reference_coupled_transport'):
        raise ValueError('E1-C paired reference identity mismatch')
    if meta['common_initial_tensor_sha256']!=cfg.get('reference_common_initial_tensor_sha256'):
        raise ValueError('E1-C reference common initialization mismatch')
    if any(base.get(k)!=cfg.get(k) for k in SIGNATURE_KEYS):
        raise ValueError('E1-C execution differs from paired reference protocol')
    return {'status':'PASS','seed':seed,**identity}


def prepare(references,output):
    from .train_e0_ready import source_digest
    from .initialization_audit import audit
    records={};signature=None;model_hashes={}
    for path in references:
        seed,base,meta,identity=reference(path)
        if seed in records:raise ValueError('Duplicate reference seed')
        current={k:base.get(k) for k in SIGNATURE_KEYS}
        if signature is None:signature=current
        if signature!=current:raise ValueError('Reference protocols differ; do not change frozen values')
        records[seed]=(Path(path).resolve(),base,meta,identity)
    if set(records)!={1337,2027,3407}:raise ValueError('Exactly three same-seed coupled reference runs required')
    for name in FROZEN_MODELS:
        previous=subprocess.check_output(['git','show',BASE_COMMIT+':research/industrial_defect_english/'+name],cwd=ROOT.parents[1])
        if (ROOT/name).read_bytes()!=previous:raise ValueError('Original model/loss changed: '+name)
        model_hashes[name]=hashlib.sha256(previous).hexdigest()
    configs={}
    for seed,(path,base,meta,identity) in sorted(records.items()):
        initial=audit(seed,['coupled_transport',*E1C_MATRIX])
        common=initial['checks']['coupled_transport']
        if initial['status']!='PASS' or common['common_tensor_collection_sha256']!=meta['common_initial_tensor_sha256']:
            raise ValueError('Reference common initialization does not match frozen factory/seed')
        transport_sha=common['transport_initial_tensor_sha256']
        historical=meta.get('transport_initial_tensor_sha256')
        if historical is not None and historical!=transport_sha:raise ValueError('Historical transport initialization mismatch')
        cfg=copy.deepcopy(base)
        cfg.update(experiment='E1_C_shared_vs_recomputed',experiment_stage='E1-C',matrix=E1C_MATRIX,
          variant='recomputed_final_transport',seed=seed,split_seed=1337,
          batch_id=f'e1c_s{seed}_{identity["executed_config_sha256"][:10]}',
          expected_source_sha256=source_digest(),scientific_protocol_id='e1c_shared_vs_recomputed_v1',
          shared_protocol_file=str(SHARED),shared_protocol_sha256=sha256(SHARED),
          reference_coupled_run_directory=str(path),reference_coupled_transport=identity,
          reference_e1a_results_commit=E1A_RESULTS_COMMIT,
          reference_common_initial_tensor_sha256=meta['common_initial_tensor_sha256'],
          reference_transport_initial_tensor_sha256=transport_sha,
          reference_transport_initialization_evidence=('HISTORICAL_HASH_AND_FROZEN_FACTORY' if historical else 'RECONSTRUCTED_FROM_BYTE_IDENTICAL_FACTORY_AND_SEED_NOT_HISTORICAL_TENSOR_RECORD'),
          frozen_model_sha256=model_hashes,scientific_gate='AUTHOR_APPROVED_SCIENTIFIC_PROTOCOL_ONLY',
          gpu_training_approved=False,scope='E1-C three new recomputed runs; coupled endpoints reused; GPU/budget approval pending')
        cfg['budget']['author_approved']=False;cfg['budget']['cap_status']='E1C_PENDING_AUTHOR_BUDGET_APPROVAL'
        if execution_protocol_errors(cfg):raise ValueError('E1-C protocol mismatch: '+str(execution_protocol_errors(cfg)))
        check_reference_binding(cfg)
        configs[f'e1c_recomputed_final_transport_s{seed}.json']=cfg
    target=Path(output);target.mkdir(parents=True,exist_ok=False)
    for name,cfg in configs.items():
        with (target/name).open('x') as f:json.dump(cfg,f,indent=2,allow_nan=False);f.write('\n')
    with (target/'SHA256SUMS.txt').open('x') as f:
        for name in configs:f.write(sha256(target/name)+'  '+name+'\n')
    return {name:sha256(target/name) for name in configs}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--references',nargs=3,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.references,a.output),indent=2))

if __name__=='__main__':main()

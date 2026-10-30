"""Derive two new configs from verified E0 metadata; never split data or train."""
import argparse,copy,hashlib,json,subprocess
from pathlib import Path
from .handoff_common import resolve_config,sha256,protocol_errors

E0_COMMIT='135d7ccfba6f112b60598dad38f5da19ac1bbc99'
FROZEN_MODELS=['models/factory.py','models/transport.py','models/losses.py','models/metrics.py','vendor/provenance.json']


def prepare(config,run_manifest,output):
    from .train_e0_ready import source_digest
    config=Path(config);meta=json.loads(Path(run_manifest).read_text())
    if meta.get('source_commit_sha')!=E0_COMMIT or meta.get('seed')!=1337:raise ValueError('Wrong E0 source/seed')
    if meta.get('result_status')!='COMPLETE' or meta.get('completed_updates')!=10000 or meta.get('planned_updates')!=10000:
        raise ValueError('A complete E0 endpoint is required')
    if sha256(config) not in (meta.get('config_sha256'),meta.get('executed_config_sha256')):raise ValueError('E0 config SHA mismatch')
    base=resolve_config(json.loads(config.read_text()),config)
    if protocol_errors(base) or base.get('seed')!=1337:raise ValueError('E0 protocol mismatch')
    if base.get('loss_protocol_id')!='neu_a2ms_mask_aware_candidate_v1' or base.get('rng_policy')!='isolated_v1':raise ValueError('E0 loss/RNG mismatch')
    for key,sha_key in [('train_manifest','train_manifest_sha256'),('validation_manifest','validation_manifest_sha256'),('quarantine_manifest','quarantine_manifest_sha256'),('condition_file','condition_sha256'),('split_audit','split_audit_sha256')]:
        if not base.get(key) or sha256(base[key])!=base.get(sha_key):raise ValueError('Frozen E0 asset mismatch: '+key)
    audit=json.loads(Path(base['split_audit']).read_text())
    if audit.get('seed')!=1337 or [audit.get(k) for k in ('actual_train','actual_validation','quarantined_train_count')]!=[2903,726,1]:
        raise ValueError('Expected original frozen E0 split: 2903/726/1; split seed 1337')
    for key,n in [('train_manifest',2903),('validation_manifest',726),('quarantine_manifest',1)]:
        if len([x for x in Path(base[key]).read_text().splitlines() if x.strip()])!=n:raise ValueError('Manifest row count mismatch: '+key)
    root=Path(__file__).resolve().parents[1];repo=root.parents[1];model_hashes={}
    for name in FROZEN_MODELS:
        prior=subprocess.check_output(['git','show',E0_COMMIT+':research/industrial_defect_english/'+name],cwd=repo)
        if (root/name).read_bytes()!=prior:raise ValueError('E0 model/loss changed: '+name)
        model_hashes[name]=hashlib.sha256(prior).hexdigest()
    shared=root/'configs/e1a_shared_protocol_v1.json'
    configs={}
    for seed in (2027,3407):
        c=copy.deepcopy(base)
        c.update(experiment='E1_A_NEU_development_only',experiment_stage='E1-A',seed=seed,split_seed=1337,
                 batch_id=f'e1a_s{seed}_{sha256(config)[:10]}',expected_source_sha256=source_digest(),
                 scientific_gate='AUTHOR_APPROVED_SCIENTIFIC_PROTOCOL_ONLY',gpu_training_approved=False,
                 reference_e0_source_commit_sha=E0_COMMIT,reference_e0_config_file_sha256=sha256(config),
                 reference_e0_config_sha256=meta.get('config_sha256'),reference_e0_executed_config_sha256=meta.get('executed_config_sha256'),
                 reference_e0_manifest_sha256=sha256(run_manifest),frozen_model_sha256=model_hashes,
                 scientific_protocol_id='e1a_multiseed_protocol_v1',
                 shared_protocol_file=str(shared),shared_protocol_sha256=sha256(shared),
                 scope='Two new training seeds only; reuse original E0 split and fixed endpoint. E1 GPU/budget approval pending.')
        c['budget']['author_approved']=False
        c['budget']['cap_status']='E1_PENDING_AUTHOR_COST_AND_RUN_APPROVAL'
        configs[f'e1a_seed{seed}.json']=c
    target=Path(output);target.mkdir(parents=True,exist_ok=False)
    for name,c in configs.items():
        with (target/name).open('x') as f:json.dump(c,f,indent=2,allow_nan=False);f.write('\n')
    with (target/'SHA256SUMS.txt').open('x') as f:
        for name in configs:f.write(sha256(target/name)+'  '+name+'\n')
    return {name:sha256(target/name) for name in configs}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--e0-config',required=True);p.add_argument('--e0-run-manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.e0_config,a.e0_run_manifest,a.output),indent=2))

if __name__=='__main__':main()

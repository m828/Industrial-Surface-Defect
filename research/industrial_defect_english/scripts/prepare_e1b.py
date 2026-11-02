"""Generate nine pending E1-B configs from real same-seed coupled endpoints."""
import argparse,copy,hashlib,json,re,subprocess
from pathlib import Path
from .handoff_common import E1B_MATRIX,resolve_config,sha256,execution_protocol_errors
from .prepare_e1a import E0_COMMIT,FROZEN_MODELS

E1A_COMMIT='e946787b58f4584268fe66478b2f4d2ccad8d344'
E1A_RESULTS_COMMIT='0d95303507017568109be95afeaf9c32a7d588c3'
ASSETS={'train_manifest':'train_manifest_sha256','validation_manifest':'validation_manifest_sha256',
        'quarantine_manifest':'quarantine_manifest_sha256','condition_file':'condition_sha256','split_audit':'split_audit_sha256'}
SIGNATURE_KEYS=[*ASSETS.values(),'optimizer','scheduler','batch_size','input_hw','optimizer_updates',
                'loss_protocol_id','loss_weights','rng_policy','initialization','augmentation','validation_every',
                'checkpoint_selection','data_governance_policy_id','stopping','study_scope','exploratory_unknown_parent_authorized']


def read_reference(root):
    root=Path(root);config=root/'executed_config.json'
    meta_path=root/'run.json' if (root/'run.json').is_file() else root/'run_manifest.json'
    meta=json.loads(meta_path.read_text());cfg=resolve_config(json.loads(config.read_text()),config)
    seed=meta.get('seed')
    if seed not in (1337,2027,3407) or cfg.get('seed')!=seed or meta.get('variant')!='coupled_transport':raise ValueError('Expected same-seed coupled reference')
    if meta.get('source_commit_sha')!=(E0_COMMIT if seed==1337 else E1A_COMMIT):raise ValueError('Reference source mismatch')
    if meta.get('result_status')!='COMPLETE' or meta.get('completed_updates')!=10000 or meta.get('planned_updates')!=10000:raise ValueError('Incomplete reference')
    if (meta.get('status') is not None and meta['status']!='complete') or meta.get('fixed_endpoint_comparable') is False:raise ValueError('Reference is not a complete endpoint')
    if sha256(config)!=meta.get('executed_config_sha256'):raise ValueError('Reference executed config SHA mismatch')
    if execution_protocol_errors(cfg) or cfg.get('split_seed',1337)!=1337:raise ValueError('Reference training protocol mismatch')
    if not re.fullmatch('[0-9a-f]{64}',meta.get('common_initial_tensor_sha256') or ''):raise ValueError('Reference initialization identity missing')
    for path,hash_key in ASSETS.items():
        if not cfg.get(path) or sha256(cfg[path])!=cfg.get(hash_key):raise ValueError('Frozen asset SHA mismatch: '+path)
    audit=json.loads(Path(cfg['split_audit']).read_text())
    if audit.get('seed')!=1337 or [audit.get(k) for k in ('actual_train','actual_validation','quarantined_train_count')]!=[2903,726,1]:raise ValueError('Wrong frozen split counts/seed')
    if audit.get('status') not in ('FROZEN','FROZEN_EXPLORATORY_D4_ONLY'):raise ValueError('Split is not frozen')
    for name,n in [('train_manifest',2903),('validation_manifest',726),('quarantine_manifest',1)]:
        if len([x for x in Path(cfg[name]).read_text().splitlines() if x.strip()])!=n:raise ValueError('Wrong manifest row count')
    return seed,cfg,meta,{'source_commit_sha':meta['source_commit_sha'],'metadata_sha256':sha256(meta_path),
                          'executed_config_sha256':sha256(config),'original_config_sha256':meta.get('config_sha256')}


def prepare(references,output):
    from .train_e0_ready import source_digest
    records={};common=None
    for path in references:
        seed,cfg,meta,identity=read_reference(path)
        if seed in records:raise ValueError('Duplicate reference seed')
        signature={k:cfg.get(k) for k in SIGNATURE_KEYS}
        if common is None:common=signature
        if signature!=common:raise ValueError('Reference frozen data/training protocols differ')
        records[seed]=(cfg,meta,identity)
    if set(records)!={1337,2027,3407}:raise ValueError('Exactly three reference seeds required')
    root=Path(__file__).resolve().parents[1];model_hashes={}
    for name in FROZEN_MODELS:
        prior=subprocess.check_output(['git','show',E1A_COMMIT+':research/industrial_defect_english/'+name],cwd=root.parents[1])
        if (root/name).read_bytes()!=prior:raise ValueError('Model/loss changed: '+name)
        model_hashes[name]=hashlib.sha256(prior).hexdigest()
    shared=root/'configs/e1b_protocol_v1.json';configs={}
    for seed,(base,meta,identity) in sorted(records.items()):
        for variant in E1B_MATRIX:
            c=copy.deepcopy(base)
            c.update(experiment='E1_B_NEU_mechanism',experiment_stage='E1-B',matrix=E1B_MATRIX,variant=variant,
                     seed=seed,split_seed=1337,batch_id=f'e1b_s{seed}_{identity["executed_config_sha256"][:10]}',
                     expected_source_sha256=source_digest(),scientific_protocol_id='e1b_mechanism_v1',
                     shared_protocol_file=str(shared),shared_protocol_sha256=sha256(shared),
                     scientific_gate='AUTHOR_APPROVED_SCIENTIFIC_PROTOCOL_ONLY',gpu_training_approved=False,
                     reference_coupled_transport=identity,reference_e1a_results_commit=E1A_RESULTS_COMMIT,
                     reference_common_initial_tensor_sha256=meta['common_initial_tensor_sha256'],
                     frozen_model_sha256=model_hashes,scope='E1-B nine new ablation runs; same-seed full models reused; budget/GPU approval pending')
            c['budget']['author_approved']=False;c['budget']['cap_status']='E1B_PENDING_AUTHOR_BUDGET_APPROVAL'
            if execution_protocol_errors(c):raise ValueError('Generated E1-B protocol mismatch')
            configs[f'e1b_{variant}_s{seed}.json']=c
    target=Path(output);target.mkdir(parents=True,exist_ok=False)
    for name,c in configs.items():
        with (target/name).open('x') as f:json.dump(c,f,indent=2,allow_nan=False);f.write('\n')
    with (target/'SHA256SUMS.txt').open('x') as f:
        for name in configs:f.write(sha256(target/name)+'  '+name+'\n')
    return {name:sha256(target/name) for name in configs}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--references',nargs=3,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.references,a.output),indent=2))

if __name__=='__main__':main()

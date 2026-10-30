"""Read-only server E0 gate. NO split creation, model forward or optimizer.step.

Writes one new report; GO_READY means ready to request training approval only.
All operational paths/config/commit are supplied explicitly. Dependencies are
queried in the current Python/container, never installed or upgraded.
"""
import argparse,json,os,platform,shutil,subprocess,sys
from pathlib import Path
from .handoff_common import git_identity,protocol_errors,resolve_config,sha256,write_new_json,verify_loaded_sources,execution_protocol_errors,verify_partition_identity


def environment():
    info={'hostname':platform.node(),'python_executable':sys.executable,'python':platform.python_version(),
          'container_marker_dockerenv':Path('/.dockerenv').exists(),
          'cgroup_identity_sha256':sha256('/proc/self/cgroup') if Path('/proc/self/cgroup').is_file() else None,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    try:
        import torch
        info.update(torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available())
        info['gpus']=[{'name':torch.cuda.get_device_name(i),'total_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    except ImportError:info['dependency_error']='torch unavailable'
    cmd=shutil.which('nvidia-smi')
    if cmd:
        r=subprocess.run([cmd,'--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
        info['gpu_usage_query']={'returncode':r.returncode,'output':r.stdout.strip()}
    else:info['gpu_usage_query']='UNAVAILABLE'
    info['scheduler_tools']={name:shutil.which(name) for name in ('sbatch','squeue')}
    return info


def verify_d4_against_source(train_list,test_list,data_root,train,val,evidence,quarantine=None,audit=None,policy='disabled'):
    """Recompute original corpus, quarantine groups and exclusions from raw files."""
    from .freeze_neu_development import scan,d4_evidence,load_families,partition_training
    original_train=scan(train_list,data_root,'train')
    original_test=scan(test_list,data_root,'test')
    actual=d4_evidence(original_train,original_test)
    if actual['train_test_pairs'] and policy=='disabled':raise ValueError('D4_cross_split_match_recomputed')
    if actual!=evidence:raise ValueError('D4_evidence_does_not_match_raw_images')
    families={}
    if audit and audit.get('lineage_evidence')=='SUPPLIED_PARENT_MAP':
        path=audit.get('family_map_path')
        if not path or sha256(path)!=audit.get('family_map_sha256'):raise ValueError('Parent map source/SHA missing or changed')
        families=load_families(path,original_train)
    partition=partition_training(original_train,actual,families,policy,
                                 audit.get('d4_matches_sha256') if audit else None)
    quarantined=quarantine or []
    if {r['sample_id'] for r in quarantined}!={r['sample_id'] for r in partition['quarantined']}:
        raise ValueError('Quarantine IDs do not match recomputed groups')
    if {r['sample_id'] for r in train+val}!={r['sample_id'] for r in partition['eligible']}:
        raise ValueError('Eligible IDs do not match recomputed groups')
    expected={r['sample_id']:r for r in partition['eligible']+partition['quarantined']}
    for row in train+val+quarantined:
        for key in ('image_sha256','mask_sha256','image_d4_canonical_sha256','group_id'):
            if row.get(key)!=expected[row['sample_id']].get(key):raise ValueError('Recomputed partition mismatch:'+key)
    for row in quarantined:
        for key in ('quarantine_policy_id','quarantine_reason','quarantine_seed_train_ids',
                    'quarantine_match_indices','d4_evidence_sha256','quarantine_matches'):
            if row.get(key)!=expected[row['sample_id']].get(key):raise ValueError('Recomputed quarantine evidence mismatch:'+key)
    if {r['group_id'] for r in train}&{r['group_id'] for r in val}:raise ValueError('D4_development_split_overlap')
    return {'train_images':len(original_train),'test_images':len(original_test),
            'cross_split_pairs':len(actual['train_test_pairs']),'quarantined_train_n':len(quarantined),
            'scope':'exact RGB D4 and supplied parent metadata only; unknown broader lineage remains UNKNOWN'}


def preflight(expected,config,require_gpu=True,variant=None,run_id=None):
    report={'stage':'P2_PREFLIGHT_ONLY','training_authorized':False,'optimizer_steps':0,
            'status':'BLOCKED','checks':{},'blockers':[]}
    try:
        ident=git_identity(expected);report['git']=ident;report['blockers']+=ident['errors']
        raw=json.loads(Path(config).read_text());cfg=resolve_config(raw,config)
        report.update(config_sha256=sha256(config),source_commit_sha=expected,variant=variant,run_id=run_id,
                      loss_protocol_id=raw.get('loss_protocol_id'),e0_protocol_id=raw.get('e0_protocol_id'),
                      data_governance_policy_id=raw.get('data_governance_policy_id'),quarantine_manifest_sha256=raw.get('quarantine_manifest_sha256'),
                      train_manifest_sha256=raw.get('train_manifest_sha256'),
                      validation_manifest_sha256=raw.get('validation_manifest_sha256'),
                      condition_sha256=raw.get('condition_sha256'),split_audit_sha256=raw.get('split_audit_sha256'))
        report['checks']['protocol_errors']=protocol_errors(cfg)
        report['checks']['execution_protocol_errors']=execution_protocol_errors(cfg)
        report['blockers']+=execution_protocol_errors(cfg)
        if variant not in cfg.get('matrix',[]):report['blockers'].append('variant_missing_or_invalid')
        expected_id=f"{cfg.get('batch_id')}:{variant}:s{cfg.get('seed')}"
        if run_id!=expected_id:report['blockers'].append('run_id_missing_or_mismatched')
        if raw.get('loss_protocol_id') not in ('e0_multiclass_detail_v1','neu_a2ms_mask_aware_candidate_v1'):
            report['blockers'].append('explicit_loss_protocol_not_approved')
        if raw.get('p3_data_freeze_approved') is not True:report['blockers'].append('P3_data_freeze_not_approved')
        if not require_gpu:report['blockers'].append('CPU_ONLY_NOT_SERVER_READY')
        # Source hashing imports no model; training modules must not be imported here.
        root=Path(__file__).resolve().parents[1]
        source=sorted(list(root.rglob('*.py'))+[root/'vendor/provenance.json'])
        import hashlib
        payload='\n'.join(str(f.relative_to(root))+':'+sha256(f) for f in source)
        source_sha=hashlib.sha256(payload.encode()).hexdigest();report['source_snapshot_sha256']=source_sha
        if raw.get('expected_source_sha256')!=source_sha:report['blockers'].append('source_snapshot_mismatch')
        report['environment']=environment()
        if require_gpu and not report['environment'].get('cuda_available'):report['blockers'].append('CUDA_unavailable_in_current_python')
        usage=report['environment'].get('gpu_usage_query')
        if require_gpu and (not isinstance(usage,dict) or usage.get('returncode')!=0):report['blockers'].append('current_GPU_usage_unverified')
        required_assets=['train_manifest','validation_manifest','condition_file','split_audit']
        if cfg.get('data_governance_policy_id')=='neu_exact_d4_quarantine_candidate_v1':required_assets.append('quarantine_manifest')
        for key in required_assets:
            path=cfg.get(key);expected_sha=raw.get('condition_sha256' if key=='condition_file' else key+'_sha256')
            if not path or not Path(path).is_file():report['blockers'].append('missing:'+key);continue
            actual=sha256(path);report['checks'][key]={'sha256':actual,'matches':actual==expected_sha}
            if actual!=expected_sha:report['blockers'].append('SHA_mismatch:'+key)
        data_root=cfg.get('data_root')
        if not data_root or not Path(data_root).is_dir():report['blockers'].append('explicit_readable_data_root_required')
        root=cfg.get('runs_root')
        report['run_root_free_bytes']=shutil.disk_usage(root).free if root and Path(root).is_dir() else None
        if report['run_root_free_bytes'] is None:report['blockers'].append('run_root_unavailable')
        elif report['run_root_free_bytes']<cfg['budget']['min_free_bytes']:report['blockers'].append('insufficient_disk')
        # P2 does not create a split. Existing frozen manifests can be read/checked.
        if all(report['checks'].get(k,{}).get('matches') for k in required_assets):
            audit=json.loads(Path(cfg['split_audit']).read_text())
            policy=cfg.get('data_governance_policy_id')
            if audit.get('data_governance_policy_id')!=policy:raise ValueError('Governance policy mismatch')
            allowed_status=['PASS_NO_CROSS_SPLIT_EXACT_D4']
            if policy=='neu_exact_d4_quarantine_candidate_v1':allowed_status.append('PASS_CROSS_SPLIT_EXACT_D4_QUARANTINED')
            if audit.get('d4_exact_check_status') not in allowed_status or audit.get('d4_transform_set')!='D4_EXACT_RGB_8_V1':
                report['blockers'].append('D4_exact_split_identity_not_cleared')
            d4_path=Path(cfg['split_audit']).parent/'d4_matches.json'
            if not d4_path.is_file() or sha256(d4_path)!=audit.get('d4_matches_sha256'):raise ValueError('D4 evidence missing/SHA mismatch')
            d4=json.loads(d4_path.read_text())
            if d4.get('schema')!='neu_d4_evidence_v1':raise ValueError('D4 schema mismatch')
            if audit.get('d4_train_test_pair_count')!=len(d4.get('train_test_pairs',[])):raise ValueError('D4 pair count mismatch')
            report['checks']['d4_matches_sha256']=sha256(d4_path)
            quarantine=[]
            if policy=='neu_exact_d4_quarantine_candidate_v1':
                if audit.get('quarantine_manifest_sha256')!=cfg.get('quarantine_manifest_sha256'):raise ValueError('Quarantine audit SHA mismatch')
                quarantine=[json.loads(x) for x in Path(cfg['quarantine_manifest']).read_text().splitlines() if x.strip()]
            repo=Path(report['git']['repo_root'])
            official_train=repo/'experiments/protocol/splits/neu_train.txt'
            official_test=repo/'experiments/protocol/splits/neu_test.txt'
            if sha256(official_train)!=audit.get('official_train_list_sha256') or sha256(official_test)!=audit.get('official_test_list_sha256'):raise ValueError('Official split source SHA mismatch')
            official_pairs={tuple(line.split()) for line in official_train.read_text().splitlines() if line.strip()}
            train=[json.loads(x) for x in Path(cfg['train_manifest']).read_text().splitlines() if x.strip()]
            val=[json.loads(x) for x in Path(cfg['validation_manifest']).read_text().splitlines() if x.strip()]
            lineage=audit.get('lineage_evidence','UNKNOWN')
            report['checks']['lineage']={'recorded_evidence':lineage,
                'status':'SUPPLIED_PARENT_METADATA' if lineage=='SUPPLIED_PARENT_MAP' else 'UNKNOWN',
                'independent_origin_verification':'NOT_ESTABLISHED_BY_THIS_SCRIPT'}
            exploratory=(cfg.get('exploratory_unknown_parent_authorized') is True and
                         cfg.get('study_scope')=='EXPLORATORY_D4_ONLY_V1' and
                         policy=='neu_exact_d4_quarantine_candidate_v1' and
                         audit.get('status')=='FROZEN_EXPLORATORY_D4_ONLY' and
                         audit.get('exploratory_unknown_parent_authorized') is True and
                         audit.get('origin_independence')=='UNKNOWN' and lineage=='UNKNOWN')
            report['checks']['exploratory_scope']='D4_ONLY_PARENT_UNKNOWN' if exploratory else 'DISABLED'
            if lineage!='SUPPLIED_PARENT_MAP' and not exploratory:report['blockers'].append('lineage_evidence_unknown_or_attestation_only')
            report['checks']['cohort']={'train_n':len(train),'validation_n':len(val),'lineage_status':audit.get('status')}
            if audit.get('status')!='FROZEN' and not exploratory:report['blockers'].append('cohort_not_frozen')
            if audit.get('actual_train')!=len(train) or audit.get('actual_validation')!=len(val):raise ValueError('Audit cohort counts mismatch')
            if audit.get('dev_train_sha256')!=raw['train_manifest_sha256'] or audit.get('dev_validation_sha256')!=raw['validation_manifest_sha256']:raise ValueError('Audit split SHA mismatch')
            if len(official_pairs)!=3630 or len([x for x in official_test.read_text().splitlines() if x.strip()])!=840:raise ValueError('Official list counts mismatch')
            report['checks']['partition_counts']=verify_partition_identity(train,val,quarantine,audit,official_pairs)
            for field in ('sample_id','group_id','image_pixel_sha256'):
                if {r[field] for r in train}&{r[field] for r in val}:report['blockers'].append('split_overlap:'+field)
            for row in train+val+quarantine:
                if (row.get('original_mask_relative'),row.get('original_image_relative')) not in official_pairs:raise ValueError('Development row not in original training list')
                if not data_root:raise ValueError('Explicit data root is missing')
                for field,relative in [('image','original_image_relative'),('mask','original_mask_relative')]:
                    if Path(row[field]).resolve()!= (Path(data_root)/row[relative]).resolve():raise ValueError('Raw path does not match official relative source')
                if row.get('original_split')!='train':raise ValueError('Official test entry in development manifest')
                for field in ('image','mask'):
                    if sha256(row[field])!=row[field+'_sha256']:raise ValueError('Raw asset SHA mismatch')
            if d4_path.is_file() and sha256(d4_path)==audit.get('d4_matches_sha256'):
                report['checks']['d4_source_recomputed']=verify_d4_against_source(official_train,official_test,data_root,train,val,d4,quarantine,audit,policy)
            # Shape/class checks use frozen paths, no prediction or evaluation.
            from PIL import Image
            import numpy as np
            counts={s:{str(c):{'positive_images':0,'gt_pixels':0} for c in range(4)} for s in ('train','validation')}
            for label,rows in [('train',train),('validation',val)]:
                for row in rows:
                    with Image.open(row['image']) as im: shape=(im.height,im.width)
                    with Image.open(row['mask']) as im:m=np.array(im)
                    if m.ndim!=2 or m.shape!=shape or not np.issubdtype(m.dtype,np.integer) or not set(np.unique(m))<={0,1,2,3}:raise ValueError('Shape/class contract failed')
                    for c in range(4):
                        n=int((m==c).sum());counts[label][str(c)]['gt_pixels']+=n;counts[label][str(c)]['positive_images']+=int(n>0)
            report['checks']['class_denominators']=counts
            if any(v['gt_pixels']==0 for v in counts['validation'].values()):report['blockers'].append('validation_missing_class')
            condition=json.loads(Path(cfg['condition_file']).read_text())
            if condition.get('train_manifest_sha256')!=raw['train_manifest_sha256']:report['blockers'].append('condition_train_identity_mismatch')
            report['checks']['official_test_usage']='EXCLUSION_ONLY_NOT_EVALUATED'
            if not audit.get('official_test_list_sha256'):report['blockers'].append('missing_official_test_exclusion_evidence')
        ledger=Path(__file__).resolve().parents[1]/'evidence/complexity_cpu.json'
        report['checks']['complexity_ledger']={'sha256':sha256(ledger),'scope':'Existing CPU conv counters, excludes elementwise and latency'} if ledger.exists() else 'MISSING_OPTIONAL_NO_SPEED_CLAIM'
        report['checks']['old_checkpoint']='OPTIONAL_FOR_FROM_SCRATCH_NOT_ACCESSED'
        from .initialization_audit import audit as initial_audit
        report['checks']['initialization_fairness']=initial_audit()
        report['source_integrity']=verify_loaded_sources(ident['repo_root'])
        if report['checks']['initialization_fairness']['status']!='PASS':report['blockers'].append('common_initialization_mismatch')
        if cfg.get('rng_policy')!='isolated_v1':report['blockers'].append('isolated_rng_policy_requires_author_confirmation')
        report['checks']['budget_and_stopping']={'policy':cfg.get('stopping',{}).get('policy_id'),'performance_early_stop':False,'response_early_stop':False,'author_approved':cfg.get('stopping',{}).get('author_approved',False)}
        budget=cfg.get('budget',{})
        if any(not isinstance(budget.get(k),(int,float)) or budget[k]<=0 for k in ('max_wall_seconds_per_run','max_batch_gpu_hours','max_run_bytes','min_free_bytes')):report['blockers'].append('concrete_budget_proposal_missing')
        report['checks']['author_approval']='STILL_REQUIRED_EVEN_IF_GO_READY'
        if not report['blockers']:report['status']='GO_READY'
    except Exception as exc:report['blockers'].append(type(exc).__name__+': '+str(exc))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--allow-cpu',action='store_true',help='unit/local checks only; cannot yield server GO_READY')
    p.add_argument('--variant',required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args()
    if Path(a.output).exists():p.error('Refuse existing report output')
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()).resolve()
    if Path(a.output).resolve().is_relative_to(repo):p.error('Report must be outside the source worktree')
    report=preflight(a.expected_commit,a.config,not a.allow_cpu,a.variant,a.run_id)
    report['execution_scope']='LOCAL_CPU_ONLY_NOT_SERVER' if a.allow_cpu else 'CURRENT_SERVER_PYTHON_PREFLIGHT'
    if a.allow_cpu:report['blockers'].append('CPU_ONLY_NOT_SERVER_READY');report['status']='BLOCKED'
    write_new_json(a.output,report);print(report['status']);return 0 if report['status']=='GO_READY' else 2

if __name__=='__main__':raise SystemExit(main())

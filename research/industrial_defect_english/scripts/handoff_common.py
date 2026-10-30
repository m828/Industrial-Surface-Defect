"""Small shared E0 handoff contracts: no model construction or training."""
import copy,hashlib,json,re,subprocess,sys
from pathlib import Path

MATRIX=['formal_a2ms','coupled_transport','capacity_control']
PATH_KEYS=['train_manifest','validation_manifest','condition_file','split_audit','runs_root','data_root','quarantine_manifest']


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)


def resolve_config(config,path):
    out=copy.deepcopy(config);base=Path(path).resolve().parent
    for key in PATH_KEYS:
        if out.get(key):out[key]=str((base/Path(out[key])).resolve())
    return out


def protocol_errors(cfg):
    want={'matrix':MATRIX,'seed':1337,'optimizer_updates':10000,'input_hw':[200,200],
          'batch_size':16,'scheduler':'constant','loss_weights':[10,1,3,3],
          'initialization':'from_scratch_common_seed','num_classes':4}
    errors=[f'protocol:{k}' for k,v in want.items() if cfg.get(k)!=v]
    if cfg.get('optimizer')!={'name':'Adam','lr':1e-4,'weight_decay':2e-6}:errors.append('protocol:optimizer')
    if cfg.get('augmentation')!='paired horizontal flip p=0.5; bilinear RGB, nearest labels; RGB/255':errors.append('protocol:augmentation')
    if cfg.get('checkpoint_selection')!='fixed final optimizer update 10000':errors.append('protocol:checkpoint_selection')
    if cfg.get('validation_every')!=1000:errors.append('protocol:validation_every')
    return errors


def execution_protocol_errors(cfg):
    """P2.7 candidate execution rules; these checks do not grant author approval."""
    errors=protocol_errors(cfg)
    for key,value in {'e0_protocol_id':'e0_fixed_endpoint_candidate_v2',
                      'loss_protocol_id':'neu_a2ms_mask_aware_candidate_v1',
                      'rng_policy':'isolated_v1'}.items():
        if cfg.get(key)!=value:errors.append('execution:'+key)
    stop=cfg.get('stopping',{})
    if stop.get('policy_id')!='fixed_endpoint_safety_only_candidate_v1':errors.append('execution:stop_policy')
    for key in ('performance_early_stop','response_early_stop'):
        if stop.get(key) is not False:errors.append('execution:'+key)
    if cfg.get('data_governance_policy_id') not in ('disabled','neu_exact_d4_quarantine_candidate_v1'):
        errors.append('execution:data_governance_policy_id')
    return errors


def verify_partition_identity(train,val,quarantine,audit,original_pairs):
    """Check sets/counts against full official source; IDs, not just row counts."""
    cohorts=(train,val,quarantine)
    ids=[{r['sample_id'] for r in rows} for rows in cohorts]
    if any(len(s)!=len(rows) for s,rows in zip(ids,cohorts)):raise ValueError('Duplicate partition sample IDs')
    if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise ValueError('Quarantine/development ID overlap')
    all_rows=train+val+quarantine
    pairs=[(r.get('original_mask_relative'),r.get('original_image_relative')) for r in all_rows]
    if len(set(pairs))!=len(pairs) or set(pairs)!=original_pairs:raise ValueError('Original training coverage not closed')
    counts={'original_train_count':len(original_pairs),'eligible_train_count':len(train)+len(val),
            'quarantined_train_count':len(quarantine),'actual_train':len(train),'actual_validation':len(val)}
    if any(audit.get(k)!=v for k,v in counts.items()):raise ValueError('Partition audit count mismatch')
    if counts['original_train_count']!=counts['eligible_train_count']+counts['quarantined_train_count']:
        raise ValueError('Partition count equation failed')
    for key,cohort_ids in [('original_train_ids_sha256',set.union(*ids)),
                           ('eligible_train_ids_sha256',ids[0]|ids[1]),
                           ('quarantined_train_ids_sha256',ids[2])]:
        fingerprint=hashlib.sha256(('\n'.join(sorted(cohort_ids))+'\n').encode()).hexdigest()
        if audit.get(key)!=fingerprint:raise ValueError('Partition ID fingerprint mismatch:'+key)
    if any(r.get('original_split')!='train' for r in all_rows):raise ValueError('Official test used in development')
    for field in ('group_id','image_d4_canonical_sha256'):
        sets=[{r[field] for r in rows} for rows in cohorts]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Group leakage:'+field)
    return counts


def verify_loaded_sources(repo_root):
    """Bind project modules and vendor AST inputs to committed bytes; fail closed."""
    root=Path(repo_root).resolve()
    prefix='research.industrial_defect_english'
    loaded={}
    for name,mod in list(sys.modules.items()):
        if name==prefix or name.startswith(prefix+'.'):
            path=getattr(mod,'__file__',None)
            if not path: continue  # namespace package
            p=Path(path).resolve()
            if not p.is_relative_to(root):raise ValueError('project_module_outside_worktree:'+name)
            loaded[name]=p
    expected=['research/industrial_defect_english/scripts/handoff_common.py',
              'research/industrial_defect_english/scripts/server_preflight.py',
              'research/industrial_defect_english/scripts/train_e0_ready.py',
              'research/industrial_defect_english/models/factory.py',
              'research/industrial_defect_english/models/losses.py',
              'research/industrial_defect_english/models/transport.py',
              'research/industrial_defect_english/models/metrics.py',
              'research/industrial_defect_english/vendor/model_resnet.py',
              'research/industrial_defect_english/vendor/model_dsmo_rs50.py',
              'research/industrial_defect_english/vendor/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py',
              'research/industrial_defect_english/vendor/loss.py',
              'research/industrial_defect_english/vendor/provenance.json']
    paths={root/x for x in expected}|set(loaded.values())
    provenance=root/'research/industrial_defect_english/vendor/provenance.json'
    vendor={x['copy']:x['sha256'] for x in json.loads(provenance.read_text())}
    allowed={'research/industrial_defect_english/vendor/'+n for n in ('model_resnet.py','model_dsmo_rs50.py','model_dsmo_rs50_eSE_adapt_detailloss_fixed.py','loss.py')}
    if set(vendor)!=allowed:raise ValueError('vendor_lock_manifest_invalid')
    checked={}
    for p in sorted(paths):
        resolved=p.resolve()
        if not resolved.is_relative_to(root):raise ValueError('source_path_outside_worktree')
        rel=resolved.relative_to(root).as_posix()
        tracked=subprocess.run(['git','ls-files','--error-unmatch','--',rel],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        if rel in vendor and not tracked:
            if p.is_symlink() or hashlib.sha256(resolved.read_bytes()).hexdigest()!=vendor[rel]:raise ValueError('external_vendor_sha_mismatch:'+rel)
            checked[rel]=vendor[rel]
            continue
        if not tracked:raise ValueError('source_not_tracked:'+rel)
        try: committed=subprocess.check_output(['git','show','HEAD:'+rel],cwd=root,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:raise ValueError('source_not_in_commit:'+rel) from exc
        actual=resolved.read_bytes()
        if actual!=committed:raise ValueError('source_differs_from_commit:'+rel)
        checked[rel]=hashlib.sha256(actual).hexdigest()
    return {'module_paths':{k:str(v) for k,v in sorted(loaded.items())},'committed_file_sha256':checked}


def bind_preflight(report_path,report_sha,fresh,expected):
    """Validate report bytes and live re-check against explicit run identity."""
    if not report_path or not report_sha:raise ValueError('preflight_report_and_sha_required')
    if not re.fullmatch('[0-9a-f]{64}',report_sha):raise ValueError('invalid_preflight_sha')
    if sha256(report_path)!=report_sha:raise ValueError('preflight_report_sha_mismatch')
    report=json.loads(Path(report_path).read_text())
    for obj in (report,fresh):
        if obj.get('status')!='GO_READY' or obj.get('blockers')!=[]:raise ValueError('preflight_not_go_ready')
        if obj.get('optimizer_steps')!=0 or obj.get('training_authorized') is not False:raise ValueError('invalid_preflight_scope')
        for key,value in expected.items():
            if obj.get(key)!=value:raise ValueError('preflight_binding_mismatch:'+key)
    # Same checked dataset identities/denominators, source files and Python context.
    for key in ('checks','source_snapshot_sha256'):
        if report.get(key)!=fresh.get(key):raise ValueError('preflight_live_recheck_mismatch:'+key)
    if not report.get('source_integrity') or not fresh.get('source_integrity'):raise ValueError('missing_source_integrity')
    for key in ('python_executable','torch','cuda_build','container_marker_dockerenv','cgroup_identity_sha256','hostname'):
        if report.get('environment',{}).get(key)!=fresh.get('environment',{}).get(key):raise ValueError('preflight_environment_mismatch:'+key)
    return report


def git_identity(expected,cwd=None):
    def git(*args):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
    if not re.fullmatch('[0-9a-f]{40}',expected):raise ValueError('Expected a full 40-character source commit SHA')
    root=git('rev-parse','--show-toplevel');head=git('rev-parse','HEAD')
    raw=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all','-z'],cwd=cwd)
    entries=raw.decode().split('\0')
    allowed={'research/industrial_defect_english/vendor/'+n for n in ('model_resnet.py','model_dsmo_rs50.py','model_dsmo_rs50_eSE_adapt_detailloss_fixed.py','loss.py')}
    dirty=[entry for entry in entries if entry and not (entry.startswith('?? ') and entry[3:] in allowed)]
    required='research/industrial_defect_english/scripts/train_e0_ready.py'
    tracked=subprocess.run(['git','ls-files','--error-unmatch',required],cwd=root,capture_output=True).returncode==0
    errors=[]
    if head!=expected:errors.append('source_commit_mismatch')
    if dirty:errors.append('dirty_worktree')
    if not tracked:errors.append('E0_source_not_tracked_in_commit')
    return {'repo_root':root,'source_commit_sha':head,'expected_commit':expected,
            'worktree_clean':not dirty,'source_tracked':tracked,'errors':errors}


def completion(status,completed,planned):
    if status=='complete' and completed==planned:return 'COMPLETE'
    return 'INCOMPLETE'

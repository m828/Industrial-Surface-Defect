"""Explicit KSDD2 execution adapter using the project's single update loop."""
import json,math,os,platform,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .handoff_common import (git_identity,verify_loaded_sources,source_snapshot_sha256,
                            sha256,bind_preflight,completion)
from .ksdd2_contract import config,protocol_errors,approvals,data_contract,KSDD2Data,P33
from .ksdd2_support import MATRIX,binary_joint_loss,binary_metrics,initialization_audit
from .e0_contract import response_stats


class Task:
    component_names=('binary_main_ohem','binary_aux_bce','binary_semantic_ohem','detail_bce_dice')
    def __init__(self,cfg):self.cfg=cfg
    def training_loss(self,model,batch,device):
        x=batch['image'].to(device);y=batch['labels'].to(device);valid=batch['valid'].to(device)
        return binary_joint_loss(model(x),y,valid,self.cfg['loss_weights'],self.cfg['loss_protocol_id'])
    @staticmethod
    def validation_summary(v):return {k:x for k,x in v.items() if k not in ('cases','transport_response')}
    @torch.no_grad()
    def validate(self,model,loader,budget_check=None,device='cuda'):
        model.eval();responses=[];handle=None;native={}
        if hasattr(model,'transport'):
            handle=model.transport.register_forward_hook(lambda m,args,out:responses.append(response_stats(args[0],args[1],out[0],out[1])))
        def cases():
            for batch in loader:
                if budget_check:budget_check()
                logits=model(batch['image'].to(device))[0]
                if logits.shape[1]!=2 or not torch.isfinite(logits).all():raise FloatingPointError('Invalid binary validation logits')
                pred=logits.argmax(1).cpu()
                for i,sid in enumerate(batch['sample_id']):
                    h,w=(int(x) for x in batch['native_hw'][i]);valid=batch['valid'][i]
                    if int(valid.sum())!=h*w or not valid[:h,:w].all():raise ValueError('Native crop/valid identity mismatch')
                    native[sid]=(h,w)
                    yield sid,batch['labels'][i,:h,:w].numpy(),pred[i,:h,:w].numpy()
        try:result=binary_metrics(cases(),small_area_threshold=P33)
        finally:
            if handle is not None:handle.remove()
        for row in result['cases']:row.update(native_h=native[row['sample_id']][0],native_w=native[row['sample_id']][1])
        result.update(dataset_id='ksdd2',evaluation_split='development_validation',coordinate_system='native_no_padding',
          transport_response=responses,response_scope='feature grid includes padding dependence; two ordered calls per recomputed forward')
        return result


def binding(expected,cfg_path,c,variant,run_id):
    return {'dataset_id':'ksdd2','task_protocol_id':c['task_protocol_id'],'data_freeze_id':c['data_freeze_id'],
      'source_commit_sha':expected,'config_sha256':sha256(cfg_path),'variant':variant,'run_id':run_id,
      'loss_protocol_id':c['loss_protocol_id'],'asset_sha256':c['asset_sha256']}


def preflight(expected,cfg_path,require_gpu=True,variant=None,run_id=None):
    from .server_preflight import environment
    r={'status':'BLOCKED','blockers':[],'checks':{},'stage':'KSDD2_PREFLIGHT_ONLY','optimizer_steps':0,'training_authorized':False}
    try:
        c=config(cfg_path);r.update(binding(expected,cfg_path,c,variant,run_id))
        identity=git_identity(expected);r['git']=identity;r['blockers']+=identity['errors']
        r['checks']['protocol_errors']=protocol_errors(c);r['blockers']+=protocol_errors(c)
        if r['checks']['protocol_errors']:raise ValueError('KSDD2 protocol contract rejected before data/output access')
        if variant not in MATRIX:raise ValueError('KSDD2 variant not in four-model whitelist')
        if run_id!=f"{c.get('batch_id')}:{variant}:s{c.get('seed')}":raise ValueError('Run ID/config mismatch')
        r['source_snapshot_sha256']=source_snapshot_sha256()
        if c.get('expected_source_sha256')!=r['source_snapshot_sha256']:r['blockers'].append('source_snapshot_mismatch')
        _,r['checks']['frozen_data']=data_contract(c,check_raw=True)
        r['checks']['initialization_fairness']=initialization_audit(c['seed'])
        if r['checks']['initialization_fairness']['status']!='PASS':r['blockers'].append('initialization_mismatch')
        r['source_integrity']=verify_loaded_sources(identity['repo_root'])
        r['environment']=environment()
        if not require_gpu:r['blockers'].append('CPU_ONLY_NOT_SERVER_READY')
        elif not r['environment'].get('cuda_available'):r['blockers'].append('CUDA_unavailable_in_current_python')
        usage=r['environment'].get('gpu_usage_query')
        if require_gpu and (not isinstance(usage,dict) or usage.get('returncode')!=0):r['blockers'].append('current_GPU_usage_unverified')
        r['checks']['approval_errors']=approvals(c);r['blockers']+=approvals(c)
        budget=c.get('budget',{})
        if any(type(budget.get(k)) not in (int,float) or not math.isfinite(budget[k]) or budget[k]<=0 for k in ('max_wall_seconds_per_run','max_batch_gpu_hours','max_run_bytes','min_free_bytes')):
            r['blockers'].append('concrete_budget_required')
        root=Path(c.get('runs_root') or '').resolve()
        if not c.get('runs_root') or not root.is_dir() or root.is_relative_to(Path(identity['repo_root']).resolve()):raise ValueError('Existing external run root required')
        if not os.access(root,os.W_OK|os.X_OK):raise ValueError('External run root is not writable')
        run=root/c['batch_id']/f'{variant}_s{c["seed"]}'
        if run.exists():r['blockers'].append('run_output_already_exists')
        for p in (root/c['batch_id']).glob('*/run.json'):
            if json.loads(p.read_text()).get('status')=='running':r['blockers'].append('batch_run_already_marked_running')
        r['run_root_free_bytes']=shutil.disk_usage(root).free
        minimum=budget.get('min_free_bytes')
        if type(minimum) in (int,float) and math.isfinite(minimum) and r['run_root_free_bytes']<minimum:r['blockers'].append('insufficient_disk')
        if not r['blockers']:r['status']='GO_READY'
    except Exception as exc:r['blockers'].append(type(exc).__name__+': '+str(exc))
    return r


def require_execution(c,args):
    errors=protocol_errors(c)+approvals(c)
    if not args.authorize_gpu:errors.append('--authorize-gpu required')
    if not args.source_commit or not args.run_id:errors.append('source commit and run ID required')
    if args.variant not in MATRIX:errors.append('wrong variant')
    if errors:raise ValueError('KSDD2 training blocked: '+str(errors))


def run(args):
    c=config(args.config)
    if args.preflight_only:
        expected=args.source_commit or git_identity_current()
        report=preflight(expected,args.config,False,args.variant,args.run_id)
        print(json.dumps(report,indent=2));return 0 if report['status']=='GO_READY' else 2
    require_execution(c,args)
    fresh=preflight(args.source_commit,args.config,True,args.variant,args.run_id)
    bind_preflight(args.preflight_report,args.preflight_sha256,fresh,binding(args.source_commit,args.config,c,args.variant,args.run_id))
    if not torch.cuda.is_available():raise ValueError('CUDA required; no real CPU-training fallback')
    cohorts,_=data_contract(c,check_raw=True);seed=c['seed'];root=Path(c['runs_root']);batch=root/c['batch_id']
    previous=[json.loads(p.read_text()) for p in batch.glob('*/run.json')]
    used=sum(r.get('elapsed_wall_seconds',0) for r in previous)
    remaining=c['budget']['max_batch_gpu_hours']*3600-used
    if remaining<=0:raise ValueError('Batch allocated-GPU-time exhausted')
    runtime_budget=dict(c['budget'],max_wall_seconds_per_run=min(c['budget']['max_wall_seconds_per_run'],remaining))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    from ..models import build_model
    from .initialization_audit import tensor_sha
    from .training_loop import execute_updates
    import hashlib,sys
    tensor_hash=lambda pairs:hashlib.sha256(str(sorted((k,tensor_sha(v)) for k,v in pairs)).encode()).hexdigest()
    expected=fresh['checks']['initialization_fairness']['checks'][args.variant]
    common=expected['common_tensor_collection_sha256']
    if args.variant!='formal_a2ms':
        reference=json.loads((batch/f'formal_a2ms_s{seed}/run.json').read_text())
        if reference.get('status')!='complete' or reference.get('result_status')!='COMPLETE' or reference.get('completed_updates')!=10000 or reference.get('planned_updates')!=10000 or reference.get('source_commit_sha')!=args.source_commit or reference.get('config_sha256')!=sha256(args.config) or reference.get('common_initial_tensor_sha256')!=common:
            raise ValueError('Complete same-source/config/seed formal_a2ms reference required')
    out=batch/f'{args.variant}_s{seed}';out.mkdir(parents=True,exist_ok=False)
    (out/'config_snapshot.json').write_bytes(Path(args.config).read_bytes())
    (out/'executed_config.json').write_text(json.dumps(c,indent=2))
    meta={'dataset_id':'ksdd2','schema':'ksdd2_run_v1','source_commit_sha':args.source_commit,'run_id':args.run_id,
      'variant':args.variant,'seed':seed,'config':c,'executed_config':c,'planned_config':json.loads(Path(args.config).read_text()),
      'config_sha256':sha256(args.config),'executed_config_sha256':sha256(out/'executed_config.json'),
      'preflight_sha256':args.preflight_sha256,'asset_sha256':c['asset_sha256'],'planned_updates':c['optimizer_updates'],
      'completed_updates':0,'status':'running','result_status':'INCOMPLETE','common_initial_tensor_sha256':None,
      'transport_initial_tensor_sha256':None,'parameter_count':None,'preflight_expected_initialization':expected,
      'python_executable':sys.executable,'python_version':platform.python_version(),'torch':torch.__version__,
      'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'runtime_budget':runtime_budget}
    (out/'run.json').write_text(json.dumps(meta,indent=2));started=time.perf_counter()
    try:
        model=build_model(args.variant,2,seed).cuda()
        actual=tensor_hash((k,v) for k,v in model.state_dict().items() if not k.startswith('transport.'))
        transport=tensor_hash((k,v) for k,v in model.state_dict().items() if k.startswith('transport.')) if hasattr(model,'transport') else None
        if actual!=common or transport!=expected['transport_initial_tensor_sha256']:raise ValueError('Actual initial tensor identity mismatch')
        meta.update(common_initial_tensor_sha256=actual,transport_initial_tensor_sha256=transport,parameter_count=sum(p.numel() for p in model.parameters()))
        (out/'run.json').write_text(json.dumps(meta,indent=2))
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        sample_rng=torch.Generator().manual_seed(seed);val_rng=torch.Generator().manual_seed(seed+1);augment_rng=random.Random(seed)
        train=DataLoader(KSDD2Data(cohorts['train'],True,augment_rng),batch_size=c['batch_size'],shuffle=True,drop_last=True,num_workers=0,generator=sample_rng)
        val=DataLoader(KSDD2Data(cohorts['validation']),batch_size=1,shuffle=False,num_workers=0,generator=val_rng)
        optimizer=torch.optim.Adam(model.parameters(),**{k:v for k,v in c['optimizer'].items() if k!='name'})
        torch.cuda.reset_peak_memory_stats()
        execute_updates(model,optimizer,train,val,c,out,meta,started,runtime_budget,sample_rng,val_rng,augment_rng,iter(train),task=Task(c))
        meta.update(status='complete',checkpoint_sha256=sha256(out/'endpoint.pt'),peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        meta.update(status='stopped' if str(exc).startswith(('BUDGET_STOP','PROTOCOL_STOP')) else 'failed',reason=repr(exc));raise
    finally:
        meta['elapsed_wall_seconds']=time.perf_counter()-started
        try:meta['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        except Exception as exc:meta['resource_query_error']=repr(exc)
        for name in ('metrics','updates'):
            if (out/(name+'.jsonl')).is_file():meta[name+'_sha256']=sha256(out/(name+'.jsonl'))
        meta['result_status']=completion(meta['status'],meta['completed_updates'],meta['planned_updates'])
        (out/'run.json').write_text(json.dumps(meta,indent=2))


def git_identity_current():
    import subprocess
    return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()

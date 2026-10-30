"""Prelaunch GPU runner; training remains explicitly blocked pending approval.

No default data/run path and no automatic real training. Explicit author gate,
GPU authorization flag, valid manifest SHA and unique output directory required.
"""
import argparse, hashlib, json, os, random, subprocess, time, shutil, copy, sys, platform
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from .handoff_common import git_identity,protocol_errors,resolve_config,completion,bind_preflight,execution_protocol_errors,verify_partition_identity
from ..models import build_model
from ..models.losses import joint_loss_with_protocol,loss_component_names
from ..models.metrics import case_metrics
from .e0_contract import require_finite,require_finite_loss,check_budget,response_stats,summarize_validation


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_digest():
    root=Path(__file__).resolve().parents[1]
    files=sorted(list(root.rglob('*.py'))+[root/'vendor/provenance.json'])
    payload='\n'.join(str(f.relative_to(root))+':'+digest(f) for f in files)
    return hashlib.sha256(payload.encode()).hexdigest()


def records(path, sha):
    if not path or not sha or digest(path)!=sha: raise ValueError('Missing or mismatched split manifest SHA')
    rows=[json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows:
        if r.get('original_split')!='train': raise ValueError('Only original training split allowed')
        for key in ['image','mask']:
            if digest(r[key])!=r.get(key+'_sha256'): raise ValueError(f'Input SHA mismatch: {r["sample_id"]}/{key}')
    if len({r['sample_id'] for r in rows})!=len(rows): raise ValueError('Duplicate sample IDs')
    return rows


class SegmentationData(Dataset):
    def __init__(self,rows,hw,augment=False,rng=None): self.rows=rows; self.hw=hw; self.augment=augment; self.rng=rng
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        from PIL import Image
        import numpy as np
        r=self.rows[i]; size=tuple(reversed(self.hw))
        im=Image.open(r['image']).convert('RGB').resize(size,Image.Resampling.BILINEAR)
        gt=Image.open(r['mask']).resize(size,Image.Resampling.NEAREST)
        if self.augment and (self.rng or random).random()<.5:
            im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT); gt=gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        x=torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float()/255
        y=torch.from_numpy(np.array(gt,copy=True)).long()
        if y.ndim!=2: raise ValueError('Expected indexed/grayscale integer labels')
        return x,y


@torch.no_grad()
def validate(model,loader,n,conditions=None,budget_check=None):
    model.eval(); matrix=torch.zeros(n,n,dtype=torch.long)
    cases=[]; normal_total=0; normal_fp=0; offset=0; responses=[]; handle=None
    if hasattr(model,'transport'):
        def diagnostic(module,inputs,outputs):
            t,beta=outputs
            responses.append(response_stats(inputs[0],inputs[1],t,beta))
        handle=model.transport.register_forward_hook(diagnostic)
    for x,y in loader:
        if budget_check: budget_check()
        logits=model(x.cuda())[0]; pred=logits.argmax(1).cpu()
        if y.min()<0 or y.max()>=n: raise ValueError('Invalid validation label')
        matrix+=torch.bincount((n*y+pred).flatten(),minlength=n*n).reshape(n,n)
        for yy,pp in zip(y,pred):
            sid=loader.dataset.rows[offset]["sample_id"]; offset+=1
            new_rows=case_metrics(yy,pp,n,sid)
            if conditions:
                for row in new_rows:
                    if not row['gt_area']: continue
                    cut=conditions['classes'][str(row['class_id'])]
                    for key,value in [('area',row['gt_fraction']),('complexity',row['inner_boundary_complexity'])]:
                        thresholds=cut[key]
                        row[key+'_group']='small' if value<=thresholds['p33'] else ('large' if value>thresholds['p67'] else 'medium')
            cases.extend(new_rows)
            if not (yy>0).any():
                normal_total+=1; normal_fp+=int((pp>0).any())
    if handle is not None: handle.remove()
    tp=matrix.diag(); union=matrix.sum(0)+matrix.sum(1)-tp
    iou=tp.float()/union.clamp_min(1)
    if (matrix.sum(1)==0).any():
        raise ValueError('E0 validation must contain GT pixels from all declared classes')
    valid=union>0
    ratio=lambda num,den:[float(a/b) if b else None for a,b in zip(num,den)]
    return {'confusion':matrix.tolist(),'iou': [float(v) if bool(ok) else None for v,ok in zip(iou,valid)],
            'miou_including_background':float(iou.mean()),'declared_class_ids':list(range(n)),
            'foreground_miou':float(iou[1:].mean()),
            'normal_images':normal_total,'normal_images_false_positive':normal_fp,
            'normal_image_fpr':normal_fp/normal_total if normal_total else None,
            'precision':ratio(tp,matrix.sum(0)),
            'recall':ratio(tp,matrix.sum(1)),
            'dice':ratio(2*tp,matrix.sum(0)+matrix.sum(1)),
            'pixel_accuracy':float(tp.sum()/matrix.sum().clamp_min(1)),
            'transport_response':responses,'cases':cases}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--variant',required=True)
    p.add_argument('--source-commit',help='full approved source commit, mandatory for training')
    p.add_argument('--run-id',help='preflight-bound batch:variant:seed identity')
    p.add_argument('--preflight-report',help='prior independently saved GO_READY report')
    p.add_argument('--preflight-sha256',help='SHA-256 of exact prior report bytes')
    p.add_argument('--authorize-gpu',action='store_true'); p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args(); planned_cfg=json.loads(Path(args.config).read_text()); cfg=resolve_config(planned_cfg,args.config)
    if protocol_errors(cfg): p.error('Fixed E0 protocol mismatch: '+str(protocol_errors(cfg)))
    if args.variant not in cfg['matrix']: p.error('variant not in predeclared matrix')
    if source_digest()!=cfg.get('expected_source_sha256'): p.error('Approved code snapshot SHA mismatch')
    if cfg['optimizer']['name']!='Adam' or cfg['scheduler']!='constant': p.error('Only declared Adam/constant supported')
    if len(cfg['loss_weights'])!=4: p.error('Four loss weights required')
    if not args.preflight_only:
        if not args.source_commit or not args.run_id: p.error('source commit and run ID required')
        if execution_protocol_errors(cfg):p.error('Candidate execution protocol mismatch: '+str(execution_protocol_errors(cfg)))
        if cfg.get('loss_protocol_id') not in ('e0_multiclass_detail_v1','neu_a2ms_mask_aware_candidate_v1'):p.error('explicit loss protocol required')
        if cfg.get('scientific_gate')!='AUTHOR_APPROVED_E0':p.error('scientific/author gate remains pending')
        from .server_preflight import preflight
        fresh=preflight(args.source_commit,args.config,True,args.variant,args.run_id)
        binding={'source_commit_sha':args.source_commit,'config_sha256':digest(args.config),
                 'variant':args.variant,'run_id':args.run_id,'loss_protocol_id':cfg['loss_protocol_id'],
                 'train_manifest_sha256':cfg.get('train_manifest_sha256'),
                 'validation_manifest_sha256':cfg.get('validation_manifest_sha256'),
                 'condition_sha256':cfg.get('condition_sha256'),'split_audit_sha256':cfg.get('split_audit_sha256'),
                 'quarantine_manifest_sha256':cfg.get('quarantine_manifest_sha256'),
                 'data_governance_policy_id':cfg.get('data_governance_policy_id'),'e0_protocol_id':cfg.get('e0_protocol_id')}
        try:bind_preflight(args.preflight_report,args.preflight_sha256,fresh,binding)
        except (ValueError,OSError,KeyError) as exc:p.error('Preflight gate: '+str(exc))
    train=records(cfg['train_manifest'],cfg['train_manifest_sha256']); val=records(cfg['validation_manifest'],cfg['validation_manifest_sha256'])
    if {r['sample_id'] for r in train}&{r['sample_id'] for r in val}: p.error('train/validation overlap')
    if {r['image_sha256'] for r in train}&{r['image_sha256'] for r in val}: p.error('train/validation content overlap')
    if len(train)<cfg['batch_size']: p.error('batch exceeds training sample count')
    if not cfg.get('condition_file') or not cfg.get('condition_sha256') or digest(cfg['condition_file'])!=cfg['condition_sha256']:
        p.error('Training-derived condition file/SHA required')
    conditions=json.loads(Path(cfg['condition_file']).read_text())
    if conditions['train_manifest_sha256']!=cfg['train_manifest_sha256']: p.error('Condition file from different train split')
    if not cfg.get('split_audit') or digest(cfg['split_audit'])!=cfg.get('split_audit_sha256'): p.error('Frozen split audit/SHA required')
    audit=json.loads(Path(cfg['split_audit']).read_text())
    exploratory=(cfg.get('exploratory_unknown_parent_authorized') is True and cfg.get('study_scope')=='EXPLORATORY_D4_ONLY_V1'
                 and cfg.get('data_governance_policy_id')=='neu_exact_d4_quarantine_candidate_v1'
                 and audit.get('status')=='FROZEN_EXPLORATORY_D4_ONLY'
                 and audit.get('exploratory_unknown_parent_authorized') is True
                 and audit.get('origin_independence')=='UNKNOWN' and audit.get('lineage_evidence')=='UNKNOWN')
    if audit['status']!='FROZEN' and not exploratory:p.error('Split not frozen')
    quarantine=[]
    if cfg.get('data_governance_policy_id')=='neu_exact_d4_quarantine_candidate_v1':
        quarantine=records(cfg.get('quarantine_manifest'),cfg.get('quarantine_manifest_sha256'))
        if audit.get('quarantine_manifest_sha256')!=cfg['quarantine_manifest_sha256']:p.error('Quarantine audit SHA mismatch')
    if audit.get('data_governance_policy_id')!=cfg.get('data_governance_policy_id'):p.error('Governance policy mismatch')
    official=Path(__file__).resolve().parents[3]/'experiments/protocol/splits/neu_train.txt'
    original_pairs={tuple(x.split()) for x in official.read_text().splitlines() if x.strip()}
    if len(original_pairs)!=3630:p.error('Original training list count mismatch')
    try:verify_partition_identity(train,val,quarantine,audit,original_pairs)
    except (KeyError,ValueError) as exc:p.error('Partition gate: '+str(exc))
    if audit['dev_train_sha256']!=cfg['train_manifest_sha256'] or audit['dev_validation_sha256']!=cfg['validation_manifest_sha256']: p.error('Split audit mismatch')
    if {r['group_id'] for r in train}&{r['group_id'] for r in val}: p.error('Family/group overlap')
    if {r['image_pixel_sha256'] for r in train}&{r['image_pixel_sha256'] for r in val}: p.error('Decoded pixel overlap')
    if set(conditions['classes'])!={'1','2','3'} or any(conditions['classes'][c][k] is None for c in conditions['classes'] for k in ('area','complexity')): p.error('Training threshold class missing')
    actual_counts={str(c):{'positive':0,'high_complexity':0} for c in (1,2,3)}
    from ..models.metrics import contour
    for _,yy in SegmentationData(val,cfg['input_hw']):
        for c in (1,2,3):
            mask=yy==c; area=int(mask.sum())
            if area:
                actual_counts[str(c)]['positive']+=1
                complexity=float(contour(mask).sum())/(area**.5)
                if complexity>conditions['classes'][str(c)]['complexity']['p67']:
                    actual_counts[str(c)]['high_complexity']+=1
    if args.preflight_only:
        print(json.dumps({'data_preflight':'PASS','actual_validation_counts':actual_counts,
                          'approval_status':cfg['scientific_gate'],'training':'NOT_EXECUTED'},indent=2)); return
    if not args.source_commit: p.error('--source-commit required for formal training')
    identity=git_identity(args.source_commit)
    if identity['errors']: p.error('Source identity gate: '+str(identity['errors']))
    if cfg.get('rng_policy')!='isolated_v1': p.error('Approved isolated RNG policy required; see reproducibility contract')
    if cfg['scientific_gate']!='AUTHOR_APPROVED_E0': p.error('scientific/author gate remains pending')
    if not cfg['budget'].get('author_approved') or not cfg['stopping'].get('author_approved'): p.error('Budget and safety-only stopping policy need author approval')
    if not args.authorize_gpu: p.error('--authorize-gpu required; no training started')
    if not torch.cuda.is_available(): p.error('CUDA unavailable; no CPU fallback training')
    if not cfg['runs_root']: p.error('runs_root must be explicitly set')
    if not Path(cfg['runs_root']).is_dir(): p.error('Existing, approved run root required')
    if shutil.disk_usage(cfg['runs_root']).free<cfg['budget']['min_free_bytes']: p.error('Insufficient approved free disk')
    if cfg.get('initialization')!='from_scratch_common_seed': p.error('This batch only supports common from-scratch initialization')
    seed=cfg['seed']; random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed); torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    batch=Path(cfg['runs_root'])/cfg['batch_id']
    used=0.
    for previous in batch.glob('*/run.json'):
        prior=json.loads(previous.read_text())
        if prior['status']=='running': p.error('An E0 batch run is already marked running; inspect before proceeding')
        used+=prior.get('elapsed_wall_seconds',0.)
    remaining=cfg['budget']['max_batch_gpu_hours']*3600-used
    if remaining<=0: p.error('Batch GPU allocation exhausted')
    runtime_budget=dict(cfg['budget'],max_wall_seconds_per_run=min(cfg['budget']['max_wall_seconds_per_run'],remaining))
    reference_meta=None
    if args.variant!='formal_a2ms':
        reference_dir=batch/f'formal_a2ms_s{seed}'
        reference_meta=json.loads((reference_dir/'run.json').read_text())
        if reference_meta.get('source_commit_sha')!=args.source_commit or reference_meta.get('config_sha256')!=digest(args.config): p.error('Reference release/config identity mismatch')
        if reference_meta['status']!='complete': p.error('Complete formal_a2ms baseline required before other batch members')
        for key in ['seed','optimizer','scheduler','batch_size','input_hw','optimizer_updates','loss_weights','loss_protocol_id','e0_protocol_id','data_governance_policy_id','quarantine_manifest_sha256','initialization','train_manifest_sha256','validation_manifest_sha256','condition_sha256','expected_source_sha256','rng_policy']:
            if reference_meta['config'].get(key)!=cfg.get(key): p.error('Reference configuration mismatch: '+key)
    run=batch/f'{args.variant}_s{seed}'
    run.mkdir(parents=True,exist_ok=False)
    metadata={'status':'running','result_status':'INCOMPLETE','completed_updates':0,'planned_updates':cfg['optimizer_updates'],
              'source_commit_sha':identity['source_commit_sha'],'run_id':args.run_id,'preflight_sha256':args.preflight_sha256,'loss_protocol_id':cfg['loss_protocol_id'],'loss_component_names':loss_component_names(cfg['loss_protocol_id']),'e0_protocol_id':cfg['e0_protocol_id'],'data_governance_policy_id':cfg['data_governance_policy_id'],'quarantine_manifest_sha256':cfg.get('quarantine_manifest_sha256'),'planned_config':planned_cfg,'executed_config':cfg,'runtime_budget':runtime_budget,'variant':args.variant,'seed':seed,'config':cfg,'config_sha256':digest(args.config),
              'python_executable':sys.executable,'python_version':platform.python_version(),
              'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
              'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'git_status':subprocess.check_output(['git','status','--short'],text=True),
              'source_files':{str(f):digest(f) for f in Path(__file__).resolve().parents[1].rglob('*.py')}}
    (run/'config_snapshot.json').write_bytes(Path(args.config).read_bytes())
    (run/'executed_config.json').write_text(json.dumps(cfg,indent=2))
    metadata['executed_config_sha256']=digest(run/'executed_config.json')
    (run/'run.json').write_text(json.dumps(metadata,indent=2))
    started=time.perf_counter()
    try:
        model=build_model(args.variant,cfg['num_classes'],seed).cuda()
        from .initialization_audit import tensor_sha
        common={k:tensor_sha(v) for k,v in model.state_dict().items() if not k.startswith('transport.')}
        metadata['common_initial_tensor_sha256']=hashlib.sha256(str(sorted(common.items())).encode()).hexdigest()
        metadata['parameter_count']=sum(p.numel() for p in model.parameters())
        if reference_meta is not None and reference_meta.get('common_initial_tensor_sha256')!=metadata['common_initial_tensor_sha256']:
            raise RuntimeError('Common initial parameter tensors differ from baseline')
        (run/'run.json').write_text(json.dumps(metadata,indent=2))
        # Constructor must not perturb the paired sampling/augmentation/GPU RNG streams.
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        sample_rng=torch.Generator().manual_seed(seed)
        val_rng=torch.Generator().manual_seed(seed+1)
        augmentation_rng=random.Random(seed)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'])
        if cfg['scheduler']!='constant': raise ValueError('This E0 runner implements constant LR only')
        train_loader=DataLoader(SegmentationData(train,cfg['input_hw'],True,rng=augmentation_rng),batch_size=cfg['batch_size'],shuffle=True,drop_last=True,num_workers=0,generator=sample_rng)
        val_loader=DataLoader(SegmentationData(val,cfg['input_hw']),batch_size=1,shuffle=False,num_workers=0,generator=val_rng)
        it=iter(train_loader)
        with (run/'metrics.jsonl').open('x') as log, (run/'updates.jsonl').open('x') as updates:
            for step in range(1,cfg['optimizer_updates']+1):
                check_budget(started,run,runtime_budget,check_disk=step%100==1)
                step_started=time.perf_counter()
                try: x,y=next(it)
                except StopIteration: it=iter(train_loader); x,y=next(it)
                model.train(); optimizer.zero_grad(set_to_none=True)
                total,parts=joint_loss_with_protocol(model(x.cuda()),y.cuda(),cfg['loss_weights'],cfg['loss_protocol_id'])
                require_finite_loss(total,parts,loss_component_names(cfg['loss_protocol_id']),run,step,metadata['completed_updates'],cfg['loss_protocol_id'])
                total.backward()
                require_finite(((n,p.grad) for n,p in model.named_parameters()),'gradient')
                optimizer.step()
                metadata['completed_updates']=step
                updates.write(json.dumps({'completed_updates':step})+'\n'); updates.flush()
                require_finite(model.named_parameters(),'parameter')
                torch.cuda.synchronize()
                step_seconds=time.perf_counter()-step_started
                row={'optimizer_updates':step,'step_seconds':step_seconds,'elapsed_wall_seconds':time.perf_counter()-started,'loss':float(total.detach()),'components':[float(v.detach()) for v in parts],'lr':optimizer.param_groups[0]['lr']}
                if step%cfg['validation_every']==0 or step==cfg['optimizer_updates']:
                    row['validation']=validate(model,val_loader,cfg['num_classes'],conditions,lambda:check_budget(started,run,runtime_budget,False))
                    row['validation_summary']=summarize_validation(row['validation'])
                    row['stop_flags']=[]  # Safety-only: performance and response remain diagnostics.
                    check_budget(started,run,runtime_budget)
                    if hasattr(model,'transport'):
                        row['transport_last_weight_norm']=float(model.transport.predict[-1].weight.detach().norm())
                    recovery={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'optimizer_updates':step,'config':cfg,
                              'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                              'cuda_rng':torch.cuda.get_rng_state_all(),'sampler_rng':sample_rng.get_state(),
                              'validation_rng':val_rng.get_state(),'augmentation_rng':augmentation_rng.getstate()}
                    torch.save(recovery,run/'recovery.tmp')
                    os.replace(run/'recovery.tmp',run/'recovery.pt')
                log.write(json.dumps(row)+'\n'); log.flush()
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'optimizer_updates':step,'config':cfg},run/'endpoint.pt')
        check_budget(started,run,runtime_budget)
        metadata.update(status='complete',checkpoint_sha256=digest(run/'endpoint.pt'),
                        elapsed_wall_seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        metadata.update(status='stopped' if str(exc).startswith(('BUDGET_STOP','PROTOCOL_STOP')) else 'failed',reason=repr(exc)); raise
    finally:
        metadata['elapsed_wall_seconds']=time.perf_counter()-started
        for name in ('metrics.jsonl','updates.jsonl'):
            if (run/name).is_file():metadata[name.replace('.jsonl','')+'_sha256']=digest(run/name)
        metadata['result_status']=completion(metadata['status'],metadata['completed_updates'],metadata['planned_updates'])
        (run/'run.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__': main()

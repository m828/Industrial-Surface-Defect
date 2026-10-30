"""Future GPU E0 runner. This round runs --help/preflight tests only.

No default data/run path and no automatic real training. Explicit author gate,
GPU authorization flag, valid manifest SHA and unique output directory required.
"""
import argparse, hashlib, json, os, random, subprocess, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from ..models import build_model
from ..models.losses import joint_loss
from ..models.metrics import case_metrics


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
    def __init__(self,rows,hw,augment=False): self.rows=rows; self.hw=hw; self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        from PIL import Image
        import numpy as np
        r=self.rows[i]; size=tuple(reversed(self.hw))
        im=Image.open(r['image']).convert('RGB').resize(size,Image.Resampling.BILINEAR)
        gt=Image.open(r['mask']).resize(size,Image.Resampling.NEAREST)
        if self.augment and random.random()<.5:
            im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT); gt=gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        x=torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float()/255
        y=torch.from_numpy(np.array(gt,copy=True)).long()
        if y.ndim!=2: raise ValueError('Expected indexed/grayscale integer labels')
        return x,y


@torch.no_grad()
def validate(model,loader,n,conditions=None):
    model.eval(); matrix=torch.zeros(n,n,dtype=torch.long)
    cases=[]; normal_total=0; normal_fp=0; offset=0; responses=[]; handle=None
    if hasattr(model,'transport'):
        def diagnostic(module,inputs,outputs):
            t,beta=outputs
            responses.append({'beta_mean':float(beta.mean()),'beta_spatial_std':float(beta.flatten(2).std(2,unbiased=False).mean()),
                              'transport_rms':float(t.square().mean().sqrt())})
        handle=model.transport.register_forward_hook(diagnostic)
    for x,y in loader:
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
    p.add_argument('--authorize-gpu',action='store_true'); p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args(); cfg=json.loads(Path(args.config).read_text())
    if not args.preflight_only:p.error('Legacy runner is diagnostic-only; formal execution requires the guarded train_e0_ready entry point')
    if args.variant not in cfg['matrix']: p.error('variant not in predeclared matrix')
    if cfg['scientific_gate']!='AUTHOR_APPROVED_E0': p.error('scientific/author gate remains pending')
    if source_digest()!=cfg.get('expected_source_sha256'): p.error('Approved code snapshot SHA mismatch')
    if cfg['optimizer']['name']!='Adam' or cfg['scheduler']!='constant': p.error('Only declared Adam/constant supported')
    if len(cfg['loss_weights'])!=4: p.error('Four loss weights required')
    train=records(cfg['train_manifest'],cfg['train_manifest_sha256']); val=records(cfg['validation_manifest'],cfg['validation_manifest_sha256'])
    if {r['sample_id'] for r in train}&{r['sample_id'] for r in val}: p.error('train/validation overlap')
    if {r['image_sha256'] for r in train}&{r['image_sha256'] for r in val}: p.error('train/validation content overlap')
    if len(train)<cfg['batch_size']: p.error('batch exceeds training sample count')
    if not cfg.get('condition_file') or not cfg.get('condition_sha256') or digest(cfg['condition_file'])!=cfg['condition_sha256']:
        p.error('Training-derived condition file/SHA required')
    conditions=json.loads(Path(cfg['condition_file']).read_text())
    if conditions['train_manifest_sha256']!=cfg['train_manifest_sha256']: p.error('Condition file from different train split')
    if args.preflight_only: print('MANIFEST_PREFLIGHT_PASS; NO_TRAINING'); return
    if not args.authorize_gpu: p.error('--authorize-gpu required; no training started')
    if not torch.cuda.is_available(): p.error('CUDA unavailable; no CPU fallback training')
    if not cfg['runs_root']: p.error('runs_root must be explicitly set')
    seed=cfg['seed']; random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed); torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    run=Path(cfg['runs_root'])/f'{cfg["experiment"]}_{args.variant}_s{seed}_{time.time_ns()}'
    run.mkdir(parents=True,exist_ok=False)
    metadata={'status':'running','variant':args.variant,'seed':seed,'config':cfg,'config_sha256':digest(args.config),
              'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
              'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'git_status':subprocess.check_output(['git','status','--short'],text=True),
              'source_files':{str(f):digest(f) for f in Path(__file__).resolve().parents[1].rglob('*.py')}}
    (run/'run.json').write_text(json.dumps(metadata,indent=2))
    try:
        model=build_model(args.variant,cfg['num_classes'],seed).cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'])
        if cfg['scheduler']!='constant': raise ValueError('This E0 runner implements constant LR only')
        train_loader=DataLoader(SegmentationData(train,cfg['input_hw'],True),batch_size=cfg['batch_size'],shuffle=True,drop_last=True,num_workers=0)
        val_loader=DataLoader(SegmentationData(val,cfg['input_hw']),batch_size=1,shuffle=False,num_workers=0)
        it=iter(train_loader)
        with (run/'metrics.jsonl').open('x') as log:
            started=time.perf_counter()
            for step in range(1,cfg['optimizer_updates']+1):
                step_started=time.perf_counter()
                try: x,y=next(it)
                except StopIteration: it=iter(train_loader); x,y=next(it)
                model.train(); optimizer.zero_grad(set_to_none=True)
                total,parts=joint_loss(model(x.cuda()),y.cuda(),cfg['loss_weights'])
                if not torch.isfinite(total): raise RuntimeError('Nonfinite loss; stop rather than retry')
                total.backward(); optimizer.step(); torch.cuda.synchronize()
                step_seconds=time.perf_counter()-step_started
                row={'optimizer_updates':step,'step_seconds':step_seconds,'elapsed_wall_seconds':time.perf_counter()-started,'loss':float(total.detach()),'components':[float(v.detach()) for v in parts],'lr':optimizer.param_groups[0]['lr']}
                if step%cfg['validation_every']==0 or step==cfg['optimizer_updates']:
                    row['validation']=validate(model,val_loader,cfg['num_classes'],conditions)
                    if hasattr(model,'transport'):
                        row['transport_last_weight_norm']=float(model.transport.predict[-1].weight.detach().norm())
                    recovery={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'optimizer_updates':step,'config':cfg,
                              'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                              'cuda_rng':torch.cuda.get_rng_state_all()}
                    torch.save(recovery,run/'recovery.tmp')
                    os.replace(run/'recovery.tmp',run/'recovery.pt')
                log.write(json.dumps(row)+'\n'); log.flush()
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'optimizer_updates':step,'config':cfg},run/'endpoint.pt')
        metadata.update(status='complete',checkpoint_sha256=digest(run/'endpoint.pt'))
    except Exception as exc:
        metadata.update(status='failed',reason=repr(exc)); raise
    finally:
        (run/'run.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__': main()

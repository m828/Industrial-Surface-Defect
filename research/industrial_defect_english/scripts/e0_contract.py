"""Explicit E0 metrics/stop contracts, independent of any model performance claim."""
import math,time,json
from pathlib import Path
import torch


def require_finite(named_tensors,kind):
    pairs=[(n,t) for n,t in named_tensors if t is not None]
    if not pairs:return
    ok=torch.stack([torch.isfinite(t).all() for _,t in pairs])
    if not ok.all():
        names=[n for (n,_),good in zip(pairs,ok.tolist()) if not good]
        raise FloatingPointError(f'Nonfinite {kind}: {names}')


def require_finite_loss(total,parts,names,root,step,completed,protocol_id):
    """Keep strict historical NaN behavior and a JSON-safe event before stopping."""
    values=[total]+list(parts);finite=[bool(torch.isfinite(x).all()) for x in values]
    if all(finite):return
    event={'event':'NONFINITE_LOSS','loss_protocol_id':protocol_id,'attempted_update':step,
           'completed_updates':completed,'result_status':'INCOMPLETE',
           'components':[{'name':name,'finite':ok,'value':float(x.detach()) if ok else None}
                         for name,x,ok in zip(['total']+list(names),values,finite)]}
    with (Path(root)/'failure_event.json').open('x') as f:json.dump(event,f,indent=2,allow_nan=False)
    raise FloatingPointError('Nonfinite loss; no formula fallback; see failure_event.json')


def check_budget(started,root,budget,check_disk=True):
    if time.perf_counter()-started>budget['max_wall_seconds_per_run']:
        raise RuntimeError('BUDGET_STOP: wall/allocated-GPU-time cap')
    if check_disk:
        size=sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
        if size>budget['max_run_bytes']:raise RuntimeError('BUDGET_STOP: run disk cap')


def response_stats(detail,semantic,t,beta):
    rms=lambda x:float(x.square().mean().sqrt())
    t_rms=rms(t)
    return {'beta_mean':float(beta.mean()),
            'beta_spatial_std':float(beta.flatten(2).std(2,unbiased=False).mean()),
            'beta_abs_q50':float(beta.abs().median()),'beta_abs_q95':float(torch.quantile(beta.abs().flatten(),.95)),
            'detail_rms':rms(detail),'difference_rms':rms(semantic-detail),
            'transport_rms':t_rms,'rho_difference':t_rms/(rms(semantic-detail)+1e-8),
            'rho_detail':t_rms/(rms(detail)+1e-8)}


def summarize_validation(validation):
    matrix=validation['confusion'];n=len(matrix)
    bg=sum(matrix[0][1:]);bg_total=sum(matrix[0])
    cross=sum(matrix[i][j] for i in range(1,n) for j in range(1,n) if i!=j)
    fg_total=sum(sum(matrix[i]) for i in range(1,n))
    out={'background_fp_pixels':bg,'background_fp_rate':bg/bg_total if bg_total else None,
         'foreground_cross_class_pixels':cross,'foreground_cross_class_rate':cross/fg_total if fg_total else None,
         'foreground_to_background_pixels':sum(matrix[i][0] for i in range(1,n)),
         'foreground_to_background_rate':sum(matrix[i][0] for i in range(1,n))/fg_total if fg_total else None,
         'class_fn_pixels':{str(i):sum(matrix[i])-matrix[i][i] for i in range(1,n)},'groups':{}}
    for c in range(1,n):
        rows=[r for r in validation['cases'] if r['class_id']==c and r['gt_area']>0]
        for axis in ('complexity','area'):
            for group in ['all','small','medium','large']:
                rr=rows if group=='all' else [r for r in rows if r.get(axis+'_group')==group]
                out['groups'][f'class{c}_{axis}_{group}']={
                    'n':len(rr),'case_macro_iou':sum(r['iou'] for r in rr)/len(rr) if rr else None,
                    'information_level':'DESCRIPTIVE_SMALL_GROUP' if len(rr)<20 else 'EXPLORATORY'}
    stats=validation.get('transport_response',[])
    out['transport']={k:sum(s[k] for s in stats)/len(stats) for k in stats[0]} if stats else None
    return out


def relative_stop_flags(current,reference,contract):
    flags=[]
    if current['miou_including_background']<reference['miou_including_background']-contract['approved_miou_drop']:
        flags.append('overall_miou_harm')
    for c,tol in contract['approved_class_iou_tolerances'].items():
        if current['iou'][int(c)]<reference['iou'][int(c)]-tol:flags.append('class_'+c+'_harm')
    a=summarize_validation(reference);b=summarize_validation(current)
    base=a['background_fp_pixels'];cand=b['background_fp_pixels']
    if (base==0 and cand>0) or (base>0 and cand>base*(1+contract['approved_background_fp_relative_increase'])):
        flags.append('background_fp_harm')
    return flags

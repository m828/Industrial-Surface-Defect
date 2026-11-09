"""Twelve KSDD2 endpoints; coupled minus three controls, paired by seed."""
import argparse,csv,json,re
from pathlib import Path
from .export_e0_artifacts import verify_package
from .handoff_common import sha256
from .ksdd2_contract import protocol_errors,FREEZE_ID,P33,COUNTS
from .ksdd2_support import MATRIX,SEEDS
from .summarize_e1a import stats

METRICS=('foreground_iou','foreground_dice','background_iou','miou_including_background','precision','recall',
 'tp','fp','fn','tn','gt_positive_macro_iou','gt_positive_macro_dice',
 'empty_gt_false_positive_images','empty_gt_image_fpr','empty_gt_false_positive_pixels','empty_gt_pixel_fpr',
 'small_positive_macro_iou','small_positive_macro_dice','other_positive_macro_iou','other_positive_macro_dice')


def summarize(packages,source_commit):
    if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('Full KSDD2 source commit required')
    runs={};signature=None;inits={};transports={};within={}
    for package in packages:
        root=Path(package);verify_package(root);read=lambda name:json.loads((root/name).read_text())
        meta=read('run_manifest.json');c=read('executed_config.json');metrics=read('metrics_summary.json')
        if meta.get('schema')!='ksdd2_light_v1' or meta.get('dataset_id')!='ksdd2' or meta.get('source_commit_sha')!=source_commit:raise ValueError('Foreign dataset/source endpoint')
        if meta.get('result_status')!='COMPLETE' or not meta.get('fixed_endpoint_comparable') or meta.get('completed_updates')!=10000 or meta.get('planned_updates')!=10000:raise ValueError('INCOMPLETE is not a KSDD2 endpoint')
        for name,key in [('executed_config.json','executed_config_sha256'),('config_snapshot.json','config_sha256')]:
            if sha256(root/name)!=meta.get(key):raise ValueError('Config SHA mismatch')
        if protocol_errors(c):raise ValueError('KSDD2 execution protocol mismatch')
        seed=meta.get('seed');variant=meta.get('variant');key=(seed,variant)
        if seed not in SEEDS or variant not in MATRIX or key in runs or c.get('seed')!=seed:raise ValueError('Duplicate/unexpected seed/model')
        if seed in within and within[seed]!=meta['config_sha256']:raise ValueError('Four conditions must share same-seed config bytes')
        within[seed]=meta['config_sha256']
        common=meta.get('common_initial_tensor_sha256')
        if not re.fullmatch('[0-9a-f]{64}',common or '') or (seed in inits and inits[seed]!=common):raise ValueError('Common initialization mismatch')
        inits[seed]=common
        if variant!='formal_a2ms':
            t=meta.get('transport_initial_tensor_sha256')
            if not re.fullmatch('[0-9a-f]{64}',t or '') or (seed in transports and transports[seed]!=t):raise ValueError('Transport initialization mismatch')
            transports[seed]=t
        keys=('asset_sha256','task_protocol_id','data_freeze_id','input_hw','mask_encoding','loss_protocol_id','loss_weights','ohem','detail_valid_radius',
              'optimizer','scheduler','batch_size','optimizer_updates','validation_every','initialization','rng_policy','augmentation','checkpoint_selection','expected_source_sha256')
        current={k:c.get(k) for k in keys}
        if signature is None:signature=current
        if signature!=current:raise ValueError('Unmatched data/protocol across seeds')
        split=read('split_identity.json')
        if split.get('data_freeze_id')!=FREEZE_ID or split.get('asset_sha256')!=c['asset_sha256'] or split.get('p33')!=P33:raise ValueError('Frozen data/area identity mismatch')
        if metrics.get('coordinate_system')!='native_no_padding' or metrics.get('evaluation_split')!='development_validation':raise ValueError('Wrong evaluation split/coordinates')
        if metrics.get('gt_positive_n')!=COUNTS['validation'][1] or metrics.get('empty_gt_n')!=COUNTS['validation'][0]-COUNTS['validation'][1]:raise ValueError('Validation denominator mismatch')
        with (root/'case_metrics.csv').open() as f:rows=list(csv.DictReader(f))
        cases={r['sample_id']:r for r in rows}
        if len(cases)!=len(rows) or len(cases)!=COUNTS['validation'][0]:raise ValueError('Incomplete or duplicated validation cases')
        with (root/'confusion_matrix.csv').open() as f:cmrows=list(csv.DictReader(f))
        cm={(int(r['gt_id']),int(r['pred_id'])):int(r['pixels']) for r in cmrows}
        if len(cmrows)!=4 or set(cm)!={(i,j) for i in range(2) for j in range(2)} or any(v<0 for v in cm.values()):raise ValueError('Not a 2x2 confusion matrix')
        tn,fp,fn,tp=(cm[0,0],cm[0,1],cm[1,0],cm[1,1])
        if any(metrics[k]!=v for k,v in dict(tn=tn,fp=fp,fn=fn,tp=tp).items()):raise ValueError('Pixel metrics/confusion mismatch')
        if abs(metrics['foreground_iou']-tp/(tp+fp+fn))>1e-12 or abs(metrics['foreground_dice']-2*tp/(2*tp+fp+fn))>1e-12:raise ValueError('Foreground metric definition mismatch')
        if sum(int(r['gt_area']) for r in rows)!=tp+fn or sum(int(r['tp']) for r in rows)!=tp:raise ValueError('Case/global pixel totals mismatch')
        runs[key]={'seed':seed,'variant':variant,'metrics':metrics,'parameter_count':meta.get('parameter_count'),
          'runtime':read('runtime_summary.json'),'response':read('mechanism_response.json'),'ohem_fallback_counts':meta.get('ohem_fallback_counts'),
          'config_sha256':meta['config_sha256'],'source_commit_sha':source_commit,'cases':cases}
    if set(runs)!={(s,m) for s in SEEDS for m in MATRIX}:raise ValueError('All four models x three seeds required')
    aggregate={};paired={}
    def summary(values):
        present=[x for x in values if x is not None]
        return dict(stats(present),missing_n=len(values)-len(present))
    for model in MATRIX:
        aggregate[model]={metric:summary([runs[s,model]['metrics'][metric] for s in SEEDS]) for metric in METRICS}
    for baseline in ('formal_a2ms','capacity_control','recomputed_final_transport'):
        differences=[]
        for seed in SEEDS:
            a=runs[seed,'coupled_transport'];b=runs[seed,baseline]
            if a['cases'].keys()!=b['cases'].keys():raise ValueError('Unpaired validation IDs')
            for sid,row in a['cases'].items():
                if any(row.get(k)!=b['cases'][sid].get(k) for k in ('gt_area','n_pixels','area_group','gt_positive','native_h','native_w')):raise ValueError('Case GT/stratum mismatch')
            differences.append({'seed':seed,**{metric:(a['metrics'][metric]-b['metrics'][metric] if a['metrics'][metric] is not None and b['metrics'][metric] is not None else None) for metric in METRICS}})
        paired[baseline]={'direction':'coupled_transport minus '+baseline,'per_seed':differences,
          'aggregate':{metric:summary([d[metric] for d in differences]) for metric in METRICS},
          'foreground_iou_direction':{'positive':sum(d['foreground_iou']>0 for d in differences),'negative':sum(d['foreground_iou']<0 for d in differences),'tied':sum(d['foreground_iou']==0 for d in differences)}}
    return {'dataset_id':'ksdd2','scope':'three independent training seeds, descriptive; not significance evidence',
      'std_definition':'sample ddof=1; undefined metrics retained as NA with missing counts',
      'error_delta_note':'positive FP/FN/normal-FPR delta means more errors in coupled',
      'aggregate':aggregate,'paired':paired,'per_run':[{k:v for k,v in runs[s,m].items() if k!='cases'} for s in SEEDS for m in MATRIX]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packages',nargs=12,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=summarize(a.packages,a.source_commit)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__':main()

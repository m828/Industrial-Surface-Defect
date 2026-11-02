"""Pair nine ablation endpoints with three existing full-model endpoints."""
import argparse,csv,json,re
from pathlib import Path
from .export_e0_artifacts import verify_package
from .handoff_common import E1B_MATRIX,sha256
from .prepare_e1b import E0_COMMIT,E1A_COMMIT,SIGNATURE_KEYS
from .summarize_e1a import stats


def read_package(path):
    root=Path(path);verify_package(root)
    read=lambda name:json.loads((root/name).read_text())
    meta=read('run_manifest.json');cfg=read('executed_config.json')
    if meta.get('result_status')!='COMPLETE' or not meta.get('fixed_endpoint_comparable') or meta.get('completed_updates')!=10000 or meta.get('planned_updates')!=10000:raise ValueError('Incomplete endpoint')
    for name,key in [('executed_config.json','executed_config_sha256'),('config_snapshot.json','config_sha256')]:
        if sha256(root/name)!=meta.get(key):raise ValueError('Endpoint config SHA mismatch')
    if cfg.get('seed')!=meta.get('seed') or cfg.get('split_seed',1337)!=1337:raise ValueError('Seed mismatch')
    with (root/'per_class_metrics.csv').open() as f:classes={int(r['class_id']):float(r['iou']) for r in csv.DictReader(f)}
    if set(classes)!={0,1,2,3}:raise ValueError('Expected four class IoUs')
    with (root/'case_metrics.csv').open() as f:
        rows=list(csv.DictReader(f));cases={(r['sample_id'],int(r['class_id'])):r for r in rows}
    if len(cases)!=len(rows):raise ValueError('Duplicate case/class rows')
    return {'meta':meta,'config':cfg,'metrics':read('metrics_summary.json'),'class_iou':classes,'cases':cases,
            'errors':read('error_decomposition.json'),'conditions':read('condition_metrics.json'),
            'response':read('response_statistics.json'),'runtime':read('runtime_summary.json'),
            'mechanism':read('mechanism_response.json') if (root/'mechanism_response.json').exists() else None,'path':str(root)}


def summarize(ablations,references,source_commit):
    if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('Full E1-B source commit required')
    refs={};runs={};signature=None;transport_initial={}
    for path,is_ref in [(p,True) for p in references]+[(p,False) for p in ablations]:
        run=read_package(path);m=run['meta'];c=run['config'];seed=m.get('seed');v=m.get('variant')
        if seed not in (1337,2027,3407):raise ValueError('Unexpected training seed')
        expected=E0_COMMIT if seed==1337 else E1A_COMMIT
        if is_ref:
            if v!='coupled_transport' or seed in refs or m.get('source_commit_sha')!=expected:raise ValueError('Invalid full-model reference')
            refs[seed]=run
        else:
            if v not in E1B_MATRIX or (seed,v) in runs or m.get('source_commit_sha')!=source_commit or c.get('experiment_stage')!='E1-B' or c.get('variant')!=v:raise ValueError('Invalid ablation identity')
            t=m.get('transport_initial_tensor_sha256')
            if not re.fullmatch('[0-9a-f]{64}',t or ''):raise ValueError('Missing initial transport identity')
            if seed in transport_initial and transport_initial[seed]!=t:raise ValueError('Transport initialization mismatch')
            transport_initial[seed]=t;runs[seed,v]=run
        s={k:c.get(k) for k in SIGNATURE_KEYS}
        if signature is None:signature=s
        if signature!=s:raise ValueError('Unmatched split/threshold/loss/training protocol')
    if set(refs)!={1337,2027,3407} or set(runs)!={(s,v) for s in refs for v in E1B_MATRIX}:raise ValueError('Need nine ablations plus three reused full endpoints')
    paired={};absolute={};per_run=[]
    absolute['coupled_transport']={metric:stats([refs[s]['metrics'][metric] for s in (1337,2027,3407)]) for metric in ('miou_including_background','foreground_miou')}
    absolute['coupled_transport']['class_iou']={c:stats([refs[s]['class_iou'][c] for s in (1337,2027,3407)]) for c in (0,1,2,3)}
    for variant in E1B_MATRIX:
        deltas=[];group_deltas={}
        for seed in (1337,2027,3407):
            a=runs[seed,variant];b=refs[seed];m=a['meta'];cfg=a['config']
            if m.get('common_initial_tensor_sha256')!=b['meta'].get('common_initial_tensor_sha256') or not m.get('common_initial_tensor_sha256'):raise ValueError('Common initialization differs from reference')
            identity=cfg.get('reference_coupled_transport',{})
            if identity.get('source_commit_sha')!=b['meta']['source_commit_sha'] or identity.get('executed_config_sha256')!=b['meta']['executed_config_sha256']:raise ValueError('Wrong paired reference config')
            if a['cases'].keys()!=b['cases'].keys():raise ValueError('Unmatched case/class IDs')
            conditional={}
            for key,row in a['cases'].items():
                prior=b['cases'][key]
                for field in ('gt_area','area_group','complexity_group'):
                    if row.get(field)!=prior.get(field):raise ValueError('Case strata/GT do not match')
                if float(row['gt_area'])<=0:continue
                d=float(row['iou'])-float(prior['iou'])
                for axis in ('area','complexity'):
                    group=f'class{key[1]}_{axis}_{row.get(axis+"_group", "ungrouped")}'
                    conditional.setdefault(group,[]).append(d)
            for g,values in conditional.items():group_deltas.setdefault(g,[]).append({'seed':seed,'n':len(values),'mean_iou_delta':sum(values)/len(values)})
            deltas.append({'seed':seed,'miou_delta':a['metrics']['miou_including_background']-b['metrics']['miou_including_background'],
                           'foreground_miou_delta':a['metrics']['foreground_miou']-b['metrics']['foreground_miou'],
                           'class_iou_delta':{c:a['class_iou'][c]-b['class_iou'][c] for c in (0,1,2,3)},
                           'error_delta':{k:a['errors'][k]-b['errors'][k] for k in a['errors'].keys() & b['errors'].keys()
                                          if type(a['errors'][k]) in (int,float) and type(b['errors'][k]) in (int,float)}})
            per_run.append({'seed':seed,'variant':variant,'errors':a['errors'],'conditions':a['conditions'],
                            'mechanism':a['mechanism'],'validation_transport_response':a['response'],'runtime':a['runtime'],
                            'reference_errors':b['errors'],'reference_runtime':b['runtime'],'config_sha256':m['config_sha256']})
        paired[variant]={'direction':'ablation minus full coupled_transport','per_seed':deltas,
                         'miou_delta':stats([d['miou_delta'] for d in deltas]),
                         'foreground_miou_delta':stats([d['foreground_miou_delta'] for d in deltas]),
                         'class_iou_delta':{c:stats([d['class_iou_delta'][c] for d in deltas]) for c in (0,1,2,3)},'paired_strata':group_deltas}
        absolute[variant]={metric:stats([runs[s,variant]['metrics'][metric] for s in (1337,2027,3407)]) for metric in ('miou_including_background','foreground_miou')}
        absolute[variant]['class_iou']={c:stats([runs[s,variant]['class_iou'][c] for s in (1337,2027,3407)]) for c in (0,1,2,3)}
    return {'scope':'same-seed ablation minus reused full model; 3 seeds descriptive, no significance claim',
            'std_definition':'sample standard deviation ddof=1','absolute':absolute,'paired':paired,'per_run':per_run}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ablation-packages',nargs=9,required=True);p.add_argument('--reference-packages',nargs=3,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=summarize(a.ablation_packages,a.reference_packages,a.source_commit)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__':main()

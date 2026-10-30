"""Summarize nine verified lightweight endpoint packages; no inference/training."""
import argparse,csv,json,math,re,statistics
from pathlib import Path
from .export_e0_artifacts import verify_package
from .handoff_common import MATRIX
from .prepare_e1a import E0_COMMIT


def stats(values):
    if not values:return {'n':0,'mean':None,'std_sample':None}
    if any(not math.isfinite(v) for v in values):raise ValueError('Nonfinite metric')
    return {'n':len(values),'mean':statistics.mean(values),'std_sample':statistics.stdev(values) if len(values)>1 else None}


def summarize(packages,e1_source_commit):
    if not re.fullmatch('[0-9a-f]{40}',e1_source_commit):raise ValueError('Full E1 source commit required')
    data={};common=None;within={};initial={}
    for package in packages:
        root=Path(package);verify_package(root)
        read=lambda name:json.loads((root/name).read_text())
        meta=read('run_manifest.json');cfg=read('executed_config.json');metrics=read('metrics_summary.json')
        if meta.get('result_status')!='COMPLETE' or not meta.get('fixed_endpoint_comparable') or meta.get('completed_updates')!=10000 or meta.get('planned_updates')!=10000:
            raise ValueError('Not a complete fixed endpoint: '+str(root))
        seed=meta.get('seed');variant=meta.get('variant');key=(seed,variant)
        if seed not in (1337,2027,3407) or variant not in MATRIX or key in data:raise ValueError('Unexpected or duplicate seed/condition')
        if cfg.get('seed')!=seed or cfg.get('split_seed',1337)!=1337:raise ValueError('Training/split seed mismatch')
        if meta.get('source_commit_sha')!=(E0_COMMIT if seed==1337 else e1_source_commit):raise ValueError('Endpoint source identity mismatch')
        if seed!=1337 and cfg.get('experiment_stage')!='E1-A':raise ValueError('Wrong E1 experiment stage')
        keys=['train_manifest_sha256','validation_manifest_sha256','condition_sha256','split_audit_sha256','quarantine_manifest_sha256',
              'optimizer_updates','optimizer','scheduler','batch_size','input_hw','initialization','loss_protocol_id','loss_weights','rng_policy','augmentation','validation_every','checkpoint_selection','data_governance_policy_id','stopping']
        signature={k:cfg.get(k) for k in keys}
        if any(not re.fullmatch('[0-9a-f]{64}',signature[k] or '') for k in keys[:5]):raise ValueError('Missing frozen split/threshold/quarantine SHA')
        if signature['loss_protocol_id']!='neu_a2ms_mask_aware_candidate_v1' or signature['rng_policy']!='isolated_v1':raise ValueError('Wrong loss/RNG')
        if common is None:common=signature
        if signature!=common:raise ValueError('Unmatched frozen data or training protocol')
        if seed in within and within[seed]!=meta.get('config_sha256'):raise ValueError('Mixed config bytes within seed')
        within[seed]=meta.get('config_sha256')
        init=meta.get('common_initial_tensor_sha256')
        if not re.fullmatch('[0-9a-f]{64}',init or ''):raise ValueError('Missing common initialization SHA')
        if seed in initial and initial[seed]!=init:raise ValueError('Unmatched common initialization within seed')
        initial[seed]=init
        with (root/'per_class_metrics.csv').open() as f:classes={int(r['class_id']):float(r['iou']) for r in csv.DictReader(f)}
        if set(classes)!={0,1,2,3}:raise ValueError('Expected four class IoUs')
        response=read('response_statistics.json')
        response_mean={k:statistics.mean(float(r[k]) for r in response) for k in response[0]} if response else None
        data[key]={'seed':seed,'variant':variant,'source_commit_sha':meta['source_commit_sha'],'config_sha256':meta['config_sha256'],
                   'miou':metrics['miou_including_background'],'foreground_miou':metrics['foreground_miou'],
                   'class_iou':classes,'errors':read('error_decomposition.json'),'conditions':read('condition_metrics.json'),
                   'response_mean':response_mean,'runtime':read('runtime_summary.json'),'package':str(root)}
    if set(data)!={(s,v) for s in (1337,2027,3407) for v in MATRIX}:raise ValueError('All nine complete endpoints required; no partial three-seed summary')
    aggregate={};paired={}
    for v in MATRIX:
        runs=[data[s,v] for s in (1337,2027,3407)]
        error_keys=set.intersection(*(set(r['errors']) for r in runs))
        groups=set.intersection(*(set(r['conditions']) for r in runs))
        aggregate[v]={'miou':stats([r['miou'] for r in runs]),'foreground_miou':stats([r['foreground_miou'] for r in runs]),
          'class_iou':{c:stats([r['class_iou'][c] for r in runs]) for c in (0,1,2,3)},
          'errors':{k:stats([r['errors'][k] for r in runs]) for k in sorted(error_keys) if all(type(r['errors'][k]) in (int,float) for r in runs)},
          'groups':{g:{'seed_n':[r['conditions'][g]['n'] for r in runs],
                       'case_macro_iou':stats([r['conditions'][g]['case_macro_iou'] for r in runs if r['conditions'][g]['case_macro_iou'] is not None])} for g in sorted(groups)}}
    for baseline in ('formal_a2ms','capacity_control'):
        deltas=[{'seed':s,'miou_delta':data[s,'coupled_transport']['miou']-data[s,baseline]['miou'],
                 'foreground_miou_delta':data[s,'coupled_transport']['foreground_miou']-data[s,baseline]['foreground_miou'],
                 'class_iou_delta':{c:data[s,'coupled_transport']['class_iou'][c]-data[s,baseline]['class_iou'][c] for c in (0,1,2,3)}} for s in (1337,2027,3407)]
        paired[baseline]={'per_seed':deltas,'miou_delta':stats([r['miou_delta'] for r in deltas]),
                          'positive_seeds':sum(r['miou_delta']>0 for r in deltas),'negative_seeds':sum(r['miou_delta']<0 for r in deltas)}
    return {'seed_count':3,'std_definition':'sample standard deviation, ddof=1','scope':'exploratory independent training seeds; no significance claim',
            'aggregate':aggregate,'paired_deltas':paired,'per_run':[data[s,v] for s in (1337,2027,3407) for v in MATRIX]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packages',nargs='+',required=True);p.add_argument('--e1-source-commit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=summarize(a.packages,a.e1_source_commit)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__':main()

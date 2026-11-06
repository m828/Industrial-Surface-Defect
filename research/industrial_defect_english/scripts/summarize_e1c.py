"""Paired shared-minus-recomputed comparison of three complete seed pairs."""
import argparse,csv,json,re
from pathlib import Path
from .handoff_common import execution_protocol_errors
from .prepare_e1b import E0_COMMIT,E1A_COMMIT,SIGNATURE_KEYS
from .prepare_e1c import FROZEN_ASSETS
from .summarize_e1a import stats
from .summarize_e1b import read_package as read_endpoint

SEEDS=(1337,2027,3407)


def response_means(rows):
    if not rows:return {'status':'MISSING_NOT_ZERO'}
    keys=set.intersection(*(set(r) for r in rows))
    return {'status':'RECORDED','n_calls':len(rows),
            'mean':{k:stats([r[k] for r in rows])['mean'] for k in sorted(keys)
                    if all(type(r[k]) in (int,float) for r in rows)},
            'aggregation':'equal forward-call means; validation batch size 1; training diagnostics kept separate'}


def read_package(path):
    run=read_endpoint(path)
    with (Path(path)/'confusion_matrix.csv').open() as f:rows=list(csv.DictReader(f))
    matrix={(int(r['gt_id']),int(r['pred_id'])):int(r['pixels']) for r in rows}
    if len(rows)!=16 or set(matrix)!={(i,j) for i in range(4) for j in range(4)} or any(v<0 for v in matrix.values()):
        raise ValueError('Expected complete nonnegative 4x4 confusion matrix')
    run['confusion']=matrix
    ids={key[0] for key in run['cases']}
    if len(ids)!=726 or set(run['cases'])!={(sid,c) for sid in ids for c in (1,2,3)}:
        raise ValueError('Expected all 726 validation images and three foreground case rows each')
    for c in (1,2,3):
        if sum(int(row['gt_area']) for (_,cc),row in run['cases'].items() if cc==c)!=sum(matrix[c,j] for j in range(4)):
            raise ValueError('Case GT pixel totals differ from confusion matrix')
    return run


def error_counts(run):
    m=run['confusion']
    return {'foreground_binary_fp_pixels':sum(m[0,j] for j in (1,2,3)),
            'foreground_binary_fn_pixels':sum(m[i,0] for i in (1,2,3)),
            'foreground_cross_class_pixels':sum(m[i,j] for i in (1,2,3) for j in (1,2,3) if i!=j),
            'per_class_fp_pixels':{str(c):sum(m[i,c] for i in range(4) if i!=c) for c in (1,2,3)},
            'per_class_fn_pixels':{str(c):sum(m[c,j] for j in range(4) if j!=c) for c in (1,2,3)}}


def summarize(recomputed,references,source_commit):
    if not re.fullmatch('[0-9a-f]{40}',source_commit):raise ValueError('Full E1-C source commit required')
    data={};signature=None
    for path,is_reference in [(p,True) for p in references]+[(p,False) for p in recomputed]:
        r=read_package(path);m=r['meta'];c=r['config'];seed=m.get('seed')
        variant='coupled_transport' if is_reference else 'recomputed_final_transport'
        key=(seed,variant)
        if seed not in SEEDS or m.get('variant')!=variant or key in data:raise ValueError('Unexpected/duplicate seed or condition')
        expected=(E0_COMMIT if seed==1337 else E1A_COMMIT) if is_reference else source_commit
        if m.get('source_commit_sha')!=expected:raise ValueError('Wrong endpoint source identity')
        if not is_reference and (c.get('experiment_stage')!='E1-C' or c.get('variant')!=variant or execution_protocol_errors(c)):
            raise ValueError('Wrong E1-C execution protocol')
        if any(c.get(k)!=v for k,v in FROZEN_ASSETS.items()):raise ValueError('Frozen data identity mismatch')
        current={k:c.get(k) for k in SIGNATURE_KEYS}
        if signature is None:signature=current
        if signature!=current:raise ValueError('Unmatched training/loss/RNG/split/threshold protocol')
        data[key]=r
    if set(data)!={(s,v) for s in SEEDS for v in ('coupled_transport','recomputed_final_transport')}:
        raise ValueError('Exactly three complete recomputed runs and three coupled references required')
    paired=[];per_run=[];absolute={};strata=[]
    for seed in SEEDS:
        shared=data[seed,'coupled_transport'];fresh=data[seed,'recomputed_final_transport']
        old=shared['meta'];new=fresh['meta'];cfg=fresh['config']
        common=new.get('common_initial_tensor_sha256');transport=new.get('transport_initial_tensor_sha256')
        if not re.fullmatch('[0-9a-f]{64}',common or '') or common!=old.get('common_initial_tensor_sha256') or common!=cfg.get('reference_common_initial_tensor_sha256'):
            raise ValueError('Common initialization mismatch')
        if not re.fullmatch('[0-9a-f]{64}',transport or '') or transport!=cfg.get('reference_transport_initial_tensor_sha256'):
            raise ValueError('Transport initialization mismatch')
        if old.get('transport_initial_tensor_sha256') not in (None,transport):raise ValueError('Historical transport initialization mismatch')
        ref=cfg.get('reference_coupled_transport',{})
        if ref.get('source_commit_sha')!=old['source_commit_sha'] or ref.get('executed_config_sha256')!=old['executed_config_sha256']:
            raise ValueError('Wrong same-seed coupled reference')
        if type(new.get('parameter_count')) is not int or new['parameter_count']!=old.get('parameter_count'):
            raise ValueError('Parameter count mismatch/missing')
        if shared['cases'].keys()!=fresh['cases'].keys():raise ValueError('Unmatched validation case/class IDs')
        groups={}
        for key,row in shared['cases'].items():
            other=fresh['cases'][key]
            if any(row.get(k)!=other.get(k) for k in ('gt_area','area_group','complexity_group')):raise ValueError('Different GT/stratum identity')
            if float(row['gt_area'])<=0:continue
            for axis in ('area','complexity'):
                group=f'class{key[1]}_{axis}_{row.get(axis+"_group", "ungrouped")}'
                groups.setdefault(group,[]).append((float(row['iou']),float(other['iou'])))
        for group,values in sorted(groups.items()):
            a=sum(v[0] for v in values)/len(values);b=sum(v[1] for v in values)/len(values)
            strata.append({'seed':seed,'group':group,'n':len(values),'coupled_case_macro_iou':a,'recomputed_case_macro_iou':b,'coupled_minus_recomputed':a-b})
        a=error_counts(shared);b=error_counts(fresh)
        paired.append({'seed':seed,'miou_delta':shared['metrics']['miou_including_background']-fresh['metrics']['miou_including_background'],
          'foreground_miou_delta':shared['metrics']['foreground_miou']-fresh['metrics']['foreground_miou'],
          'class_iou_delta':{str(c):shared['class_iou'][c]-fresh['class_iou'][c] for c in range(4)},
          'error_count_delta':{k:a[k]-b[k] for k in a if type(a[k]) is int},
          'direction':'coupled_transport minus recomputed_final_transport; positive error delta means more errors in coupled'})
        mechanism=fresh['mechanism']
        if not mechanism or mechanism.get('call_order')!=['stage1','final'] or mechanism.get('shared_predictor') is not True:
            raise ValueError('Missing E1-C stage-separated response identity')
        stage_rows=mechanism.get('validation_by_stage',{})
        if not stage_rows.get('stage1') or len(stage_rows['stage1'])!=len(stage_rows.get('final',[])):
            raise ValueError('Missing/unequal stage response calls')
        if fresh['response']!=[x for pair in zip(stage_rows['stage1'],stage_rows['final']) for x in pair]:
            raise ValueError('Stage response order differs from raw validation response')
        for variant,r in [('coupled_transport',shared),('recomputed_final_transport',fresh)]:
            per_run.append({'seed':seed,'variant':variant,'metrics':r['metrics'],'class_iou':r['class_iou'],
              'confusion_matrix':[[r['confusion'][i,j] for j in range(4)] for i in range(4)],
              'errors':error_counts(r),'error_rates_and_conditions':{'errors':r['errors'],'conditions':r['conditions']},
              'parameter_count':r['meta']['parameter_count'],'runtime':r['runtime'],
              'source_commit_sha':r['meta']['source_commit_sha'],'config_sha256':r['meta']['config_sha256'],
              'executed_config_sha256':r['meta']['executed_config_sha256'],
              'response':({'stage1':response_means(r['response']),'final':'same T reused, no second predictor call'} if variant=='coupled_transport' else
                          {phase:response_means(stage_rows[phase]) for phase in ('stage1','final')}),
              'training_response':r['mechanism']})
    for variant in ('coupled_transport','recomputed_final_transport'):
        rr=[data[s,variant] for s in SEEDS]
        absolute[variant]={k:stats([r['metrics'][k] for r in rr]) for k in ('miou_including_background','foreground_miou')}
        absolute[variant]['class_iou']={str(c):stats([r['class_iou'][c] for r in rr]) for c in range(4)}
    return {'scope':'Three same-seed pairs on frozen development validation; descriptive, no significance claim',
      'comparison_direction':'coupled_transport minus recomputed_final_transport','std_definition':'sample ddof=1',
      'absolute':absolute,'paired':paired,'paired_aggregate':{k:stats([p[k] for p in paired]) for k in ('miou_delta','foreground_miou_delta')},
      'paired_strata':strata,'per_run':per_run}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recomputed-packages',nargs=3,required=True);p.add_argument('--reference-packages',nargs=3,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=summarize(a.recomputed_packages,a.reference_packages,a.source_commit)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__':main()

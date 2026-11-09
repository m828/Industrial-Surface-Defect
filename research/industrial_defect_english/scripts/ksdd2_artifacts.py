"""KSDD2 lightweight schema; reuse existing content/size/checksum restrictions."""
import csv,json,re
from pathlib import Path
from .handoff_common import sha256,completion
from .ksdd2_contract import data_contract,FREEZE_ID,P33
from .ksdd2_support import MATRIX


def export(source,output,preflight_report=None):
    from .export_e0_artifacts import check_content,verify_package,ALLOWED
    source=Path(source);out=Path(output)
    if out.exists():raise FileExistsError('Refuse existing KSDD2 export')
    m=json.loads((source/'run.json').read_text());c=m.get('executed_config',{})
    if m.get('dataset_id')!='ksdd2' or c.get('dataset_id')!='ksdd2' or m.get('variant') not in MATRIX:raise ValueError('Not a KSDD2 run')
    check_content(m);gaps=[];status=completion(m.get('status'),m.get('completed_updates'),m.get('planned_updates'))
    for name,key in [('config_snapshot.json','config_sha256'),('executed_config.json','executed_config_sha256'),
                     ('updates.jsonl','updates_sha256'),('metrics.jsonl','metrics_sha256'),('endpoint.pt','checkpoint_sha256')]:
        p=source/name
        if not p.is_file() or not m.get(key):gaps.append('MISSING:'+name)
        elif sha256(p)!=m[key]:raise ValueError('Run artifact SHA mismatch: '+name)
    if (source/'updates.jsonl').is_file():
        updates=[json.loads(x)['completed_updates'] for x in (source/'updates.jsonl').read_text().splitlines()]
        if updates!=list(range(1,m.get('completed_updates',0)+1)):gaps.append('UPDATE_JOURNAL_MISMATCH')
    pf=None
    if preflight_report:
        pf=json.loads(Path(preflight_report).read_text())
        if sha256(preflight_report)!=m.get('preflight_sha256') or pf.get('status')!='GO_READY' or pf.get('dataset_id')!='ksdd2' or pf.get('source_commit_sha')!=m.get('source_commit_sha') or pf.get('config_sha256')!=m.get('config_sha256') or pf.get('asset_sha256')!=c.get('asset_sha256'):
            gaps.append('PREFLIGHT_BINDING_MISMATCH')
    else:gaps.append('MISSING_PREFLIGHT')
    rows=[json.loads(x) for x in (source/'metrics.jsonl').read_text().splitlines()] if (source/'metrics.jsonl').exists() else []
    last=next((x for x in reversed(rows) if 'validation' in x),None)
    comparable=bool(status=='COMPLETE' and m.get('completed_updates')==m.get('planned_updates')==10000 and
      last and last['optimizer_updates']==10000 and re.fullmatch('[0-9a-f]{40}',m.get('source_commit_sha','')) and not gaps)
    cohorts,identity=data_contract(c,check_raw=False)
    out.mkdir(parents=True,exist_ok=False)
    def j(name,value):
        check_content(value)
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    def table(name,fields,values):
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
    manifest={k:m.get(k) for k in ('dataset_id','variant','seed','source_commit_sha','run_id','config_sha256',
      'executed_config_sha256','preflight_sha256','checkpoint_sha256','planned_updates','completed_updates',
      'common_initial_tensor_sha256','transport_initial_tensor_sha256','parameter_count','reason','ohem_fallback_counts')}
    manifest.update(schema='ksdd2_light_v1',result_status=status,fixed_endpoint_comparable=comparable,provenance_gaps=gaps,
                    data_freeze_id=FREEZE_ID,asset_sha256=c['asset_sha256'])
    j('run_manifest.json',manifest)
    for name in ('config_snapshot.json','executed_config.json'):(out/name).write_bytes((source/name).read_bytes())
    if pf is not None:(out/'preflight_report.json').write_bytes(Path(preflight_report).read_bytes())
    else:j('preflight_report.json',{'status':'MISSING'})
    j('split_identity.json',dict(identity,p33=P33))
    for split in ('train','validation'):
        with (out/f'dev_{split}_ids.jsonl').open('x') as f:
            for row in cohorts[split]:f.write(json.dumps({k:row.get(k) for k in ('sample_id','group_id','image_sha256','mask_sha256','gt_positive','native_hw')})+'\n')
    j('environment_snapshot.json',{k:m.get(k) for k in ('python_executable','python_version','torch','cuda','gpu','source_commit_sha')})
    j('runtime_summary.json',{k:m.get(k) for k in ('completed_updates','planned_updates','elapsed_wall_seconds','peak_cuda_bytes','parameter_count','runtime_budget')})
    if last:
        v=last['validation']
        if v.get('dataset_id')!='ksdd2' or v.get('coordinate_system')!='native_no_padding' or v.get('evaluation_split')!='development_validation':raise ValueError('Wrong KSDD2 validation coordinates/split')
        summary={k:value for k,value in v.items() if k not in ('cases','transport_response')}
        j('metrics_summary.json',dict(summary,fixed_endpoint_comparable=comparable,result_status=status))
        matrix=v['confusion_matrix']
        if len(matrix)!=2 or any(len(r)!=2 for r in matrix):raise ValueError('Expected binary confusion matrix')
        table('confusion_matrix.csv',['gt_id','pred_id','pixels'],[{'gt_id':i,'pred_id':k,'pixels':matrix[i][k]} for i in range(2) for k in range(2)])
        ratio=lambda a,b:a/b if b else None
        classes=[]
        for i in range(2):
            tp=matrix[i][i];fn=sum(matrix[i])-tp;fp=sum(r[i] for r in matrix)-tp
            classes.append(dict(class_id=i,iou=ratio(tp,tp+fp+fn),dice=ratio(2*tp,2*tp+fp+fn),precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn)))
        table('per_class_metrics.csv',['class_id','iou','dice','precision','recall'],classes)
        j('error_decomposition.json',{k:summary[k] for k in ('tp','fp','fn','tn','empty_gt_n','empty_gt_false_positive_images','empty_gt_image_fpr','empty_gt_false_positive_pixels','empty_gt_pixel_fpr')})
        j('condition_metrics.json',{k:value for k,value in summary.items() if k.startswith(('small_','other_','gt_positive_'))})
        response=v.get('transport_response',[]);j('response_statistics.json',response)
        calls=0 if m['variant']=='formal_a2ms' else (2 if m['variant']=='recomputed_final_transport' else 1)
        if comparable and len(response)!=calls*len(cohorts['validation']):raise ValueError('Response call count mismatch')
        j('mechanism_response.json',{'calls_per_image':calls,'scope':'FEATURE_GRID_DIAGNOSTIC_INCLUDES_PADDING_DEPENDENCE',
          'stage1':response[::calls] if calls else [],'final_recomputed':response[1::2] if calls==2 else [],
          'final_reuses_stage1':m['variant']=='coupled_transport','training_last_batch':last.get('training_transport_response',[])})
        cases=v['cases'];table('case_metrics.csv',sorted({k for r in cases for k in r}),cases)
    else:j('metrics_summary.json',{'result_status':status,'fixed_endpoint_comparable':False,'status':'NO_VALIDATION'})
    for p in out.iterdir():
        if p.name not in ALLOWED or p.stat().st_size>4_000_000:raise ValueError('KSDD2 export whitelist/size violation')
    if sum(p.stat().st_size for p in out.iterdir())>15_000_000:raise ValueError('KSDD2 export exceeds package cap')
    (out/'artifact_checksums.sha256').write_text(''.join(sha256(p)+'  '+p.name+'\n' for p in sorted(out.iterdir())))
    verify_package(out);return manifest

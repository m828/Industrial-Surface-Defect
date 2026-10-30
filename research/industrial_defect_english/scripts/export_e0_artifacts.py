"""Export only small structured E0 evidence; never checkpoints/data/log blobs.

A stopped run remains INCOMPLETE; its last diagnostic is never a fixed endpoint.
This tool does not commit/push and must be followed by author/content review.
"""
import argparse,csv,hashlib,json,re
from pathlib import Path
from .handoff_common import MATRIX,completion,sha256
from .e0_contract import summarize_validation

ALLOWED={'run_manifest.json','preflight_report.json','config_snapshot.json','executed_config.json',
 'environment_snapshot.json','metrics_summary.json','per_class_metrics.csv','confusion_matrix.csv',
 'error_decomposition.json','response_statistics.json','runtime_summary.json','case_metrics.csv',
 'condition_metrics.json','split_identity.json','dev_train_ids.jsonl','dev_validation_ids.jsonl',
 'artifact_checksums.sha256'}
SECRET_KEYS={'password','passwd','token','api_key','access_token','private_key','secret'}


def check_content(obj):
    if isinstance(obj,dict):
        if any((str(k).lower() in SECRET_KEYS or re.search(r'password|passwd|private[_-]?key|access[_-]?token|github[_-]?token|api[_-]?key|authorization',str(k),re.I)) and v not in (None,'') for k,v in obj.items()):raise ValueError('Sensitive key/value in export')
        for v in obj.values():check_content(v)
    elif isinstance(obj,list):
        for v in obj:check_content(v)
    elif isinstance(obj,str):
        if re.search(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|(?:https?|ssh)://[^\s/]+:[^\s/@]+@|Bearer [A-Za-z0-9._-]+|gh[pousr]_[A-Za-z0-9]{30,}',obj):
            raise ValueError('Private-key/credential-URL shape in export')


def verify_package(path):
    root=Path(path)
    for p in root.iterdir():
        if not p.is_file() or p.is_symlink() or p.name not in ALLOWED:raise ValueError('Non-whitelist artifact: '+p.name)
        if p.stat().st_size>4_000_000:raise ValueError('Artifact exceeds small-file cap')
    required={'run_manifest.json','preflight_report.json','config_snapshot.json','executed_config.json',
              'environment_snapshot.json','runtime_summary.json','metrics_summary.json','split_identity.json','artifact_checksums.sha256'}
    if not required<={p.name for p in root.iterdir()}:raise ValueError('Missing required artifact interface')
    if sum(p.stat().st_size for p in root.iterdir())>15_000_000:raise ValueError('Package exceeds total small-result cap')
    for p in root.iterdir():
        if p.suffix in ('.json','.jsonl'):
            if p.suffix=='.json':check_content(json.loads(p.read_text()))
            else:
                for line in p.read_text().splitlines():check_content(json.loads(line))
        elif p.suffix=='.csv':
            for row in csv.DictReader(p.read_text().splitlines()):check_content(row)
    manifest=root/'artifact_checksums.sha256'
    names=set()
    for line in manifest.read_text().splitlines():
        h,name=line.split('  ',1)
        if name not in ALLOWED or name in names or Path(name).name!=name:raise ValueError('Unsafe/duplicate artifact name')
        names.add(name)
        if sha256(root/name)!=h:raise ValueError('Artifact SHA mismatch: '+name)
    if names!={p.name for p in root.iterdir()}-{'artifact_checksums.sha256'}:raise ValueError('Incomplete checksum manifest')
    return True


def export(run_dir,output,preflight_report=None):
    source=Path(run_dir);target=Path(output)
    if target.exists():raise FileExistsError('Refuse existing export directory')
    meta=json.loads((source/'run.json').read_text())
    check_content(meta)
    if meta.get('variant') not in MATRIX:raise ValueError('Unknown E0 condition')
    rows=[json.loads(x) for x in (source/'metrics.jsonl').read_text().splitlines()] if (source/'metrics.jsonl').exists() else []
    planned=meta.get('planned_updates',meta.get('config',{}).get('optimizer_updates'))
    actual=meta.get('completed_updates')
    # Legacy log-derived update counts are marked as such, not invented metadata.
    status=completion(meta.get('status'),actual,planned) if actual is not None else 'INCOMPLETE_UNKNOWN_UPDATES'
    last=next((r for r in reversed(rows) if 'validation' in r),None)
    cfg=meta.get('executed_config',meta.get('config',{}))
    provenance=[]
    for key in ('train_manifest_sha256','validation_manifest_sha256','condition_sha256','split_audit_sha256','expected_source_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',cfg.get(key) or ''):provenance.append('MISSING_PROTOCOL_ID:'+key)
    if not preflight_report:provenance.append('MISSING_PREFLIGHT_REPORT')
    else:
        pf=json.loads(Path(preflight_report).read_text())
        if pf.get('status')!='GO_READY' or pf.get('config_sha256')!=meta.get('config_sha256') or pf.get('git',{}).get('source_commit_sha')!=meta.get('source_commit_sha'):
            provenance.append('PREFLIGHT_IDENTITY_MISMATCH')
    for filename,key in [('config_snapshot.json','config_sha256'),('executed_config.json','executed_config_sha256'),
                         ('metrics.jsonl','metrics_sha256'),('updates.jsonl','updates_sha256'),('endpoint.pt','checkpoint_sha256')]:
        p=source/filename;expected=meta.get(key)
        if not p.is_file() or not expected:provenance.append('MISSING:'+filename)
        elif sha256(p)!=expected:raise ValueError('Executed evidence SHA mismatch: '+filename)
    if (source/'updates.jsonl').exists():
        updates=[json.loads(x)['completed_updates'] for x in (source/'updates.jsonl').read_text().splitlines()]
        if updates!=list(range(1,(actual or 0)+1)):provenance.append('UPDATE_JOURNAL_MISMATCH')
    endpoint=bool(status=='COMPLETE' and last and last['optimizer_updates']==actual==planned and
                  re.fullmatch('[0-9a-f]{40}',meta.get('source_commit_sha') or '') and not provenance)

    target.mkdir(parents=True,exist_ok=False)
    def j(name,obj):
        check_content(obj)
        with (target/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
    def csvfile(name,fields,records):
        with (target/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    selected={k:meta.get(k) for k in ('variant','seed','source_commit_sha','config_sha256','executed_config_sha256',
               'planned_updates','completed_updates','reason','checkpoint_sha256','metrics_sha256','updates_sha256','common_initial_tensor_sha256','parameter_count')}
    selected.update(result_status=status,fixed_endpoint_comparable=endpoint,provenance_gaps=provenance,
                    updates_identity='METADATA' if actual is not None else 'MISSING_NOT_INFERRED_FROM_PLAN')
    j('run_manifest.json',selected)
    for name in ('config_snapshot.json','executed_config.json'):
        p=source/name
        if p.exists():
            obj=json.loads(p.read_text());check_content(obj)
            # Byte-for-byte snapshot hash remains reproducible.
            (target/name).write_bytes(p.read_bytes())
        else:j(name,{'status':'MISSING_LEGACY_SNAPSHOT'})
    cfg=meta.get('executed_config',meta.get('config',{}))
    split={k:cfg.get(k) for k in ('train_manifest_sha256','validation_manifest_sha256','condition_sha256','split_audit_sha256')}
    for kind in ('train','validation'):
        p=cfg.get(kind+'_manifest')
        if p and Path(p).is_file():
            if sha256(p)!=split[kind+'_manifest_sha256']:raise ValueError('Split changed since execution')
            records=[json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
            safe_keys=('sample_id','group_id','original_split','image_sha256','mask_sha256','image_pixel_sha256','original_image_relative')
            with (target/('dev_'+kind+'_ids.jsonl')).open('x') as f:
                for row in records:f.write(json.dumps({k:row.get(k) for k in safe_keys})+'\n')
            split[kind+'_view_sha256']=sha256(target/('dev_'+kind+'_ids.jsonl'))
        else:split[kind+'_view_status']='MISSING'
    split['scope']='ID/hash views, not byte-identical execution manifests; original SHAs remain separate'
    j('split_identity.json',split)
    if preflight_report:j('preflight_report.json',json.loads(Path(preflight_report).read_text()))
    else:j('preflight_report.json',{'status':'MISSING_NOT_RECONSTRUCTED'})
    j('environment_snapshot.json',{k:meta.get(k) for k in ('python_executable','python_version','torch','cuda','gpu','git_head','source_commit_sha')})
    j('runtime_summary.json',{'result_status':status,'completed_updates':actual,'planned_updates':planned,
        'elapsed_wall_seconds':meta.get('elapsed_wall_seconds'),'peak_cuda_bytes':meta.get('peak_cuda_bytes'),
        'validation_row_update':last['optimizer_updates'] if last else None})
    if last:
        v=last['validation'];summary=summarize_validation(v)
        j('metrics_summary.json',{'result_status':status,'fixed_endpoint_comparable':endpoint,
          'scope':'FIXED_ENDPOINT' if endpoint else 'LAST_DIAGNOSTIC_NOT_ENDPOINT',
          'miou_including_background':v['miou_including_background'],'foreground_miou':v['foreground_miou']})
        fields=['class_id','iou','precision','recall','dice']
        csvfile('per_class_metrics.csv',fields,[dict(class_id=i,**{k:v.get(k,[None]*len(v['iou']))[i] for k in fields[1:]}) for i in range(len(v['iou']))])
        csvfile('confusion_matrix.csv',['gt_id','pred_id','pixels'],[{'gt_id':i,'pred_id':k,'pixels':x} for i,row in enumerate(v['confusion']) for k,x in enumerate(row)])
        j('error_decomposition.json',{k:value for k,value in summary.items() if k not in ('groups','transport')})
        j('condition_metrics.json',summary['groups']);j('response_statistics.json',v.get('transport_response',[]))
        cases=v.get('cases',[]);fields=sorted({key for row in cases for key in row}) or ['sample_id','class_id']
        csvfile('case_metrics.csv',fields,cases)
    else:
        for name in ('metrics_summary.json','error_decomposition.json','condition_metrics.json','response_statistics.json'):
            j(name,{'status':'MISSING_NO_VALIDATION','result_status':status})
    for p in target.iterdir():
        if p.name not in ALLOWED or p.stat().st_size>4_000_000:raise ValueError('Export whitelist/size violation')
    if sum(p.stat().st_size for p in target.iterdir())>15_000_000:raise ValueError('Package exceeds 15MB cap')
    (target/'artifact_checksums.sha256').write_text(''.join(sha256(p)+'  '+p.name+'\n' for p in sorted(target.iterdir())))
    verify_package(target);return selected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir');p.add_argument('--output',required=True)
    p.add_argument('--preflight-report');p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.verify:verify_package(a.output)
    else:
        if not a.run_dir:p.error('--run-dir required for export')
        export(a.run_dir,a.output,a.preflight_report)

if __name__=='__main__':main()

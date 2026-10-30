"""Read-only paired E0 endpoint comparison; never select a different checkpoint."""
import argparse,json
from pathlib import Path
from .e0_contract import summarize_validation


def read_endpoint(run):
    root=Path(run);meta=json.loads((root/'run.json').read_text())
    if meta.get('result_status')!='COMPLETE' or meta.get('completed_updates')!=meta.get('planned_updates'):
        raise ValueError('INCOMPLETE or unknown update identity cannot be an endpoint')
    if meta['status']!='complete':raise ValueError('Endpoint comparison requires complete run, not best-so-far')
    from .handoff_common import sha256
    for filename,key in [('config_snapshot.json','config_sha256'),('executed_config.json','executed_config_sha256'),('metrics.jsonl','metrics_sha256'),('updates.jsonl','updates_sha256'),('endpoint.pt','checkpoint_sha256')]:
        if not (root/filename).is_file() or sha256(root/filename)!=meta.get(key):raise ValueError('Unverified endpoint evidence: '+filename)
    rows=[json.loads(x) for x in (root/'metrics.jsonl').read_text().splitlines()]
    row=rows[-1]
    if row['optimizer_updates']!=meta['config']['optimizer_updates'] or 'validation' not in row:
        raise ValueError('Not declared final endpoint')
    return meta,row


def compare(a,b):
    ma,ra=read_endpoint(a);mb,rb=read_endpoint(b)
    if ma.get('source_commit_sha')!=mb.get('source_commit_sha'):raise ValueError('Different source commits')
    from .handoff_common import execution_protocol_errors
    for meta in (ma,mb):
        if execution_protocol_errors(meta['config']):raise ValueError('Endpoint does not use the candidate execution protocol')
    if ma.get('config_sha256')!=mb.get('config_sha256'):raise ValueError('Different config bytes')
    for k in ['seed','train_manifest_sha256','validation_manifest_sha256','condition_sha256','optimizer_updates',
              'initialization','optimizer','scheduler','input_hw','batch_size','loss_weights','loss_protocol_id',
              'e0_protocol_id','data_governance_policy_id','quarantine_manifest_sha256','augmentation','validation_every',
              'checkpoint_selection','stopping','expected_source_sha256','rng_policy']:
        if ma['config'].get(k)!=mb['config'].get(k):raise ValueError(f'Unmatched protocol {k}')
    va=ra['validation'];vb=rb['validation']
    ca={(r['sample_id'],r['class_id']):r for r in va['cases']};cb={(r['sample_id'],r['class_id']):r for r in vb['cases']}
    if ca.keys()!=cb.keys():raise ValueError('Different validation sample/class IDs')
    groups={}
    for c in (1,2,3):
        for group in ['all','small','medium','large']:
            keys=[k for k,r in ca.items() if r['class_id']==c and r['gt_area']>0 and
                  (group=='all' or r.get('complexity_group')==group)]
            delta=[cb[k]['iou']-ca[k]['iou'] for k in keys]
            groups[f'class{c}_complexity_{group}']={'n':len(delta),'mean_iou_delta':sum(delta)/len(delta) if delta else None,
                'positive':sum(d>1e-12 for d in delta),'negative':sum(d< -1e-12 for d in delta),'tied':sum(abs(d)<=1e-12 for d in delta),
                'single_case_equivalent_tolerance_proposal':1/len(delta) if len(delta)>=20 else None}
    return {'base':ma['variant'],'candidate':mb['variant'],
            'miou_delta':vb['miou_including_background']-va['miou_including_background'],
            'foreground_miou_delta':vb['foreground_miou']-va['foreground_miou'],
            'class_iou_delta':[y-x for x,y in zip(va['iou'],vb['iou'])],
            'base_summary':summarize_validation(va),'candidate_summary':summarize_validation(vb),'paired_groups':groups,
            'decision':'AUTHOR_REVIEW_REQUIRED; single seed exploratory, no significance or automatic GO'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-run',required=True);p.add_argument('--candidate-run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=compare(a.base_run,a.candidate_run)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2)

if __name__=='__main__':main()

"""E1 seed/config/output checks using synthetic metadata only; no training."""
import copy,csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from ..scripts.handoff_common import protocol_errors,sha256
from ..scripts.initialization_audit import audit
from ..scripts.prepare_e1a import prepare,E0_COMMIT
from ..scripts.summarize_e1a import summarize,stats


class E1Tests(unittest.TestCase):
    def base(self):return json.loads(Path('research/industrial_defect_english/configs/e0_exploratory_d4_runtime_template_v1.json').read_text())

    def test_only_training_seed_changes(self):
        cfg=self.base()
        for seed in (2027,3407):
            c=dict(cfg,experiment_stage='E1-A',seed=seed,split_seed=1337)
            self.assertEqual(protocol_errors(c),[])
            self.assertIn('protocol:split_seed',protocol_errors(dict(c,split_seed=seed)))
        self.assertIn('protocol:seed',protocol_errors(dict(cfg,seed=2027)))
        self.assertIn('protocol:seed',protocol_errors(dict(cfg,experiment_stage='E1-A',seed=1337,split_seed=1337)))

    def test_common_initialization_per_new_seed(self):
        hashes=[]
        for seed in (2027,3407):
            result=audit(seed);self.assertEqual(result['status'],'PASS');self.assertEqual(result['seed'],seed)
            self.assertEqual(len({v['common_tensor_collection_sha256'] for v in result['checks'].values()}),1)
            hashes.append(result['checks']['formal_a2ms']['common_tensor_collection_sha256'])
        self.assertNotEqual(*hashes)

    def test_prepare_preserves_frozen_assets_and_pending_approval(self):
        with tempfile.TemporaryDirectory(prefix='e1 config path ') as td:
            root=Path(td);base=self.base()
            for key,count in [('train_manifest',2903),('validation_manifest',726),('quarantine_manifest',1)]:
                p=root/key;p.write_text('{}\n'*count);base[key]=str(p);base[key+'_sha256']=sha256(p)
            for key,obj in [('condition_file',{'train_manifest_sha256':base['train_manifest_sha256']}),
                            ('split_audit',{'seed':1337,'actual_train':2903,'actual_validation':726,'quarantined_train_count':1})]:
                p=root/key;p.write_text(json.dumps(obj));base[key]=str(p);base['condition_sha256' if key=='condition_file' else key+'_sha256']=sha256(p)
            cfg=root/'e0.json';cfg.write_text(json.dumps(base));before=cfg.read_bytes()
            meta=root/'run.json';meta.write_text(json.dumps({'seed':1337,'source_commit_sha':E0_COMMIT,'result_status':'COMPLETE','completed_updates':10000,'planned_updates':10000,'executed_config_sha256':sha256(cfg)}))
            result=prepare(cfg,meta,root/'new')
            self.assertEqual(len(result),2);self.assertEqual(cfg.read_bytes(),before)
            batches=[]
            for seed in (2027,3407):
                c=json.loads((root/f'new/e1a_seed{seed}.json').read_text());self.assertEqual(c['seed'],seed);self.assertEqual(c['split_seed'],1337)
                for key in ('train_manifest_sha256','validation_manifest_sha256','quarantine_manifest_sha256','condition_sha256','split_audit_sha256'):self.assertEqual(c[key],base[key])
                self.assertFalse(c['budget']['author_approved']);self.assertFalse(c['gpu_training_approved'])
                self.assertNotEqual(c['scientific_gate'],'AUTHOR_APPROVED_E0');self.assertEqual(c['optimizer_updates'],10000)
                batches.append(c['batch_id'])
                self.assertEqual(c['reference_e0_config_file_sha256'],sha256(cfg))
            self.assertNotEqual(*batches)
            with self.assertRaises(FileExistsError):prepare(cfg,meta,root/'new')
            (root/'train_manifest').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen E0 asset'):prepare(cfg,meta,root/'bad')

    def test_summary_nine_endpoints_and_missing_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);paths=[]
            cfg=self.base();cfg.update({k:'b'*64 for k in ('train_manifest_sha256','validation_manifest_sha256','condition_sha256','split_audit_sha256','quarantine_manifest_sha256')})
            for i,seed in enumerate((1337,2027,3407)):
                for v in ('formal_a2ms','capacity_control','coupled_transport'):
                    p=root/f'{v}_{seed}';p.mkdir();paths.append(p)
                    c=dict(cfg,seed=seed,split_seed=1337)
                    if seed!=1337:c['experiment_stage']='E1-A'
                    meta={'seed':seed,'variant':v,'source_commit_sha':E0_COMMIT if seed==1337 else 'a'*40,'result_status':'COMPLETE','fixed_endpoint_comparable':True,'completed_updates':10000,'planned_updates':10000,'config_sha256':str(i)*64,'common_initial_tensor_sha256':'c'*64}
                    value=.5+i*.01+(.02 if v=='coupled_transport' else 0)
                    for name,obj in [('run_manifest.json',meta),('executed_config.json',c),('metrics_summary.json',{'miou_including_background':value,'foreground_miou':value}),('error_decomposition.json',{'foreground_to_background_pixels':5}),('condition_metrics.json',{'class1_area_small':{'n':20,'case_macro_iou':.5}}),('response_statistics.json',[]),('runtime_summary.json',{'elapsed_wall_seconds':2})]:
                        (p/name).write_text(json.dumps(obj))
                    (p/'per_class_metrics.csv').write_text('class_id,iou\n0,0.9\n1,0.5\n2,0.5\n3,0.5\n')
            with patch('research.industrial_defect_english.scripts.summarize_e1a.verify_package',return_value=True):
                s=summarize(paths,'a'*40);self.assertEqual(s['seed_count'],3)
                self.assertAlmostEqual(s['aggregate']['formal_a2ms']['miou']['std_sample'],.01)
                self.assertAlmostEqual(s['paired_deltas']['formal_a2ms']['miou_delta']['mean'],.02)
                self.assertEqual(s['paired_deltas']['capacity_control']['positive_seeds'],3)
                with self.assertRaises(ValueError):summarize(paths[:-1],'a'*40)
                with self.assertRaises(ValueError):summarize(paths,'d'*40)
            self.assertAlmostEqual(stats([.1,.2,.3])['std_sample'],.1)

if __name__=='__main__':unittest.main()

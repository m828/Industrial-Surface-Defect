"""P1/P2 tests use temporary synthetic records only; no real data or GPU."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from ..scripts.handoff_common import protocol_errors,resolve_config,completion,write_new_json,git_identity
from ..scripts.export_e0_artifacts import export,verify_package,check_content


def fixture(root,status='stopped',completed=7):
    root.mkdir()
    cfg={'optimizer_updates':10000}
    (root/'config_snapshot.json').write_text(json.dumps(cfg));(root/'executed_config.json').write_text(json.dumps(cfg))
    meta={'status':status,'variant':'coupled_transport','completed_updates':completed,'planned_updates':10000,
          'source_commit_sha':'a'*40,'config_sha256':hashlib.sha256((root/'config_snapshot.json').read_bytes()).hexdigest(),
          'executed_config':cfg,'executed_config_sha256':hashlib.sha256((root/'executed_config.json').read_bytes()).hexdigest(),'seed':1337}
    (root/'run.json').write_text(json.dumps(meta))
    v={'confusion':[[10,1,0,0],[1,5,0,0],[0,0,6,0],[0,0,0,7]],'iou':[.8,.7,1.,1.],
       'precision':[.9,.8,1.,1.],'recall':[.9,.8,1.,1.],'dice':[.9,.8,1.,1.],
       'miou_including_background':.875,'foreground_miou':.9,'cases':[], 'transport_response':[]}
    (root/'metrics.jsonl').write_text(json.dumps({'optimizer_updates':completed,'validation':v})+'\n')


class HandoffTests(unittest.TestCase):
    def test_protocol_is_locked(self):
        p=Path('research/industrial_defect_english/configs/e0_neu_prelaunch.json')
        c=json.loads(p.read_text());self.assertEqual(protocol_errors(c),[])
        for key,value in [('seed',1),('optimizer_updates',9999),('batch_size',8),('loss_weights',[1,1,1,1])]:
            changed=copy.deepcopy(c);changed[key]=value;self.assertTrue(protocol_errors(changed))
    def test_config_bytes_and_relative_path(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'config.json';p.write_bytes(b'{"train_manifest":"split.jsonl"}\n')
            before=p.read_bytes();c=json.loads(before);r=resolve_config(c,p)
            self.assertEqual(r['train_manifest'],str(Path(td)/'split.jsonl'))
            self.assertEqual(c['train_manifest'],'split.jsonl');self.assertEqual(before,p.read_bytes())
    def test_completion_not_plan(self):
        self.assertEqual(completion('stopped',7000,10000),'INCOMPLETE')
        self.assertEqual(completion('complete',9999,10000),'INCOMPLETE')
        self.assertEqual(completion('complete',10000,10000),'COMPLETE')
    def test_new_output_only(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'report.json';write_new_json(p,{'status':'synthetic'})
            with self.assertRaises(FileExistsError):write_new_json(p,{})
    def test_full_commit_required(self):
        with self.assertRaises(ValueError):git_identity('main')
    def test_incomplete_export_and_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root/'run')
            result=export(root/'run',root/'small')
            self.assertEqual(result['result_status'],'INCOMPLETE');self.assertFalse(result['fixed_endpoint_comparable'])
            self.assertTrue(verify_package(root/'small'))
            with self.assertRaises(FileExistsError):export(root/'run',root/'small')
    def test_whitelist_and_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root/'run');export(root/'run',root/'small')
            (root/'small'/'weights.pt').write_text('synthetic forbidden artifact')
            with self.assertRaises(ValueError):verify_package(root/'small')
            (root/'small'/'weights.pt').unlink()
            (root/'small'/'runtime_summary.json').write_text('{}')
            with self.assertRaises(ValueError):verify_package(root/'small')
    def test_complete_missing_or_mismatched_evidence_not_endpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root/'run','complete',10000)
            result=export(root/'run',root/'missing')
            self.assertFalse(result['fixed_endpoint_comparable'])
            (root/'run'/'config_snapshot.json').write_text('{}')
            with self.assertRaises(ValueError):export(root/'run',root/'bad_sha')
    def test_empty_or_sensitive_self_signed_package_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);root.joinpath('artifact_checksums.sha256').write_text('')
            with self.assertRaises(ValueError):verify_package(root)
        with self.assertRaises(ValueError):check_content({'github_token':'SYNTHETIC_NOT_REAL'})
        with self.assertRaises(ValueError):check_content({'Authorization':'Bearer SYNTHETIC'})

    def test_secret_keys_rejected(self):
        with self.assertRaises(ValueError):check_content({'access_token':'SYNTHETIC_NOT_A_REAL_CREDENTIAL'})
    def test_no_training_call_in_preflight(self):
        import ast
        p=Path('research/industrial_defect_english/scripts/server_preflight.py')
        tree=ast.parse(p.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ('step','backward') for n in calls))
        self.assertNotIn('train_e0_ready',p.read_text())
    def test_preflight_blocked_without_training(self):
        from ..scripts.server_preflight import preflight
        from unittest.mock import patch
        import subprocess,torch
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        cfg='research/industrial_defect_english/configs/e0_neu_prelaunch.json'
        with patch('research.industrial_defect_english.scripts.server_preflight.environment',return_value={'cuda_available':False}), patch.object(torch.optim.SGD,'step',side_effect=AssertionError('NO OPTIMIZER')):
            result=preflight(head,cfg,False)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertEqual(result['optimizer_steps'],0)
        self.assertIn('CPU_ONLY_NOT_SERVER_READY',result['blockers'])

    def test_common_tensors_and_sampler_rng(self):
        from ..scripts.initialization_audit import audit
        result=audit();self.assertEqual(result['status'],'PASS')
        self.assertEqual(len({v['common_tensor_collection_sha256'] for v in result['checks'].values()}),1)
        import random,torch
        from torch.utils.data import DataLoader,TensorDataset
        ds=TensorDataset(torch.arange(32))
        def order(noise):
            gen=torch.Generator().manual_seed(1337)
            random.seed(noise);torch.manual_seed(noise)
            return torch.cat([x[0] for x in DataLoader(ds,batch_size=4,shuffle=True,generator=gen)]).tolist()
        self.assertEqual(order(1),order(999))
        # Exercise the actual augmentation class, not a mirrored random call.
        from ..scripts.train_e0_ready import SegmentationData
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);im=root/'x.png';mask=root/'y.png'
            Image.fromarray(np.tile(np.arange(20,dtype=np.uint8)%4,(20,1))).save(mask)
            Image.fromarray(np.zeros((20,20,3),dtype=np.uint8)).save(im)
            row={'image':str(im),'mask':str(mask)}
            a=SegmentationData([row],[20,20],True,rng=random.Random(1337))
            b=SegmentationData([row],[20,20],True,rng=random.Random(1337))
            aa=[a[0][1] for _ in range(6)]
            random.seed(999);[random.random() for _ in range(100)]
            bb=[b[0][1] for _ in range(6)]
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(aa,bb)))
            self.assertTrue(any(not torch.equal(aa[0],x) for x in aa[1:]))

if __name__=='__main__':unittest.main()

"""CPU-only synthetic fail-closed contract tests; no optimizer or real data."""
import copy,hashlib,json,tempfile,unittest,types,sys
from pathlib import Path
from unittest.mock import patch
from ..scripts.handoff_common import bind_preflight,protocol_errors,verify_loaded_sources


class P26GateTests(unittest.TestCase):
    def setUp(self):
        self.expected={'source_commit_sha':'a'*40,'config_sha256':'b'*64,'variant':'formal_a2ms',
          'run_id':'synthetic:formal_a2ms:s1337','loss_protocol_id':'neu_a2ms_mask_aware_candidate_v1',
          'train_manifest_sha256':'c'*64,'validation_manifest_sha256':'d'*64,
          'condition_sha256':'e'*64,'split_audit_sha256':'f'*64}
        self.report=dict(self.expected,status='GO_READY',blockers=[],optimizer_steps=0,
          training_authorized=False,checks={'cohort':'synthetic'},source_snapshot_sha256='1'*64,
          source_integrity={'synthetic':'test'},environment={'python_executable':'synthetic'})

    def check(self,old=None,fresh=None,sha_override=None):
        with tempfile.TemporaryDirectory(prefix='e0 path with spaces ') as td:
            p=Path(td)/'report.json';p.write_text(json.dumps(old if old is not None else self.report))
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            return bind_preflight(p,sha_override or h,fresh if fresh is not None else self.report,self.expected)

    def test_synthetic_binding_pass(self):
        self.assertEqual(self.check()['status'],'GO_READY')

    def test_missing_or_damaged_report(self):
        with self.assertRaises((ValueError,OSError)):bind_preflight(None,'a'*64,self.report,self.expected)
        with self.assertRaises(ValueError):self.check(sha_override='0'*64)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'report';p.write_text('{bad')
            with self.assertRaises(ValueError):bind_preflight(p,hashlib.sha256(p.read_bytes()).hexdigest(),self.report,self.expected)

    def test_all_bound_identities_and_blocked(self):
        for field in self.expected:
            for old_or_new in ('old','fresh'):
                changed=copy.deepcopy(self.report);changed[field]='incorrect'
                with self.subTest(field=field,object=old_or_new),self.assertRaises(ValueError):
                    self.check(**{old_or_new:changed})
        for change in ({'status':'BLOCKED'},{'blockers':['bad']},{'optimizer_steps':1},{'training_authorized':True},
                       {'checks':{'cohort':'forged'}},{'source_integrity':None}):
            bad=dict(self.report,**change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.check(old=bad)

    def test_protocol_rejects_different_augmentation_or_endpoint(self):
        p=Path('research/industrial_defect_english/configs/e0_handoff_template.json')
        cfg=json.loads(p.read_text());self.assertEqual(protocol_errors(cfg),[])
        for key in ('augmentation','checkpoint_selection','validation_every'):
            c=dict(cfg,**{key:'different'})
            with self.subTest(key=key):self.assertIn('protocol:'+key,protocol_errors(c))

    def test_module_outside_worktree_rejected(self):
        fake=types.ModuleType('research.industrial_defect_english.evil')
        fake.__file__='/outside/other_source.py'
        with patch.dict(sys.modules,{fake.__name__:fake}),self.assertRaisesRegex(ValueError,'outside_worktree'):
            verify_loaded_sources(Path.cwd())

    def test_worktree_sources_untracked_rejected(self):
        from ..scripts import handoff_common
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with patch.object(handoff_common.sys,'modules',{}),patch.object(handoff_common.subprocess,'run',return_value=types.SimpleNamespace(returncode=1)):
                with self.assertRaises((ValueError,FileNotFoundError)):verify_loaded_sources(root)

    def test_modified_tracked_module_bytes_rejected(self):
        from ..scripts import handoff_common
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'research/industrial_defect_english/models/factory.py'
            p.parent.mkdir(parents=True);p.write_bytes(b'altered')
            manifest=root/'research/industrial_defect_english/vendor/provenance.json';manifest.parent.mkdir(parents=True)
            manifest.write_bytes(Path('research/industrial_defect_english/vendor/provenance.json').read_bytes())
            fake=types.ModuleType('research.industrial_defect_english.models.factory');fake.__file__=str(p)
            # Only synthetic Git responses: no repository/index or commit is created.
            with patch.object(handoff_common.sys,'modules',{fake.__name__:fake}),\
                 patch.object(handoff_common.subprocess,'run',return_value=types.SimpleNamespace(returncode=0)),\
                 patch.object(handoff_common.subprocess,'check_output',return_value=b'committed other bytes'):
                with self.assertRaisesRegex(ValueError,'source_differs_from_commit'):verify_loaded_sources(root)

    def test_blocked_preflight_does_not_reach_optimizer(self):
        import torch
        from ..scripts import train_e0_ready
        cfg=json.loads(Path('research/industrial_defect_english/configs/e0_handoff_template.json').read_text())
        cfg.update(loss_protocol_id='neu_a2ms_mask_aware_candidate_v1',scientific_gate='AUTHOR_APPROVED_E0')
        cfg['expected_source_sha256']='synthetic'
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'config.json';p.write_text(json.dumps(cfg))
            args=['runner','--config',str(p),'--variant','formal_a2ms','--source-commit','a'*40,
                  '--run-id','synthetic:formal_a2ms:s1337','--authorize-gpu']
            with patch.object(sys,'argv',args),patch.object(train_e0_ready,'source_digest',return_value='synthetic'),\
                 patch('research.industrial_defect_english.scripts.server_preflight.preflight',return_value={'status':'BLOCKED'}),\
                 patch.object(torch.optim.Adam,'step',side_effect=AssertionError('optimizer called')) as step:
                with self.assertRaises(SystemExit):train_e0_ready.main()
                step.assert_not_called()

    def test_preflight_recomputes_d4_not_just_claimed_empty_pairs(self):
        import numpy as np
        from PIL import Image
        from ..scripts.freeze_neu_development import scan,d4_evidence
        from ..scripts.server_preflight import verify_d4_against_source
        with tempfile.TemporaryDirectory(prefix='d4 space ') as td:
            root=Path(td);(root/'images').mkdir();(root/'masks').mkdir()
            a=np.arange(48,dtype=np.uint8).reshape(4,4,3);b=np.full((4,4,3),120,dtype=np.uint8)
            for name,img in [('a',a),('b',b),('test',np.rot90(a,2))]:
                Image.fromarray(img).save(root/'images'/f'{name}.png')
                Image.fromarray(np.zeros((4,4),dtype=np.uint8)).save(root/'masks'/f'{name}.png')
            tr=root/'train.txt';tr.write_text('masks/a.png images/a.png\nmasks/b.png images/b.png\n')
            te=root/'test.txt';te.write_text('masks/test.png images/test.png\n')
            rows=scan(tr,root,'train');tests=scan(te,root,'test');evidence=d4_evidence(rows,tests)
            forged=dict(evidence,train_test_pairs=[])
            with self.assertRaisesRegex(ValueError,'D4_cross_split_match_recomputed'):
                verify_d4_against_source(tr,te,root,rows[:1],rows[1:],forged)

    def test_training_guards_precede_optimizer_step(self):
        text=Path('research/industrial_defect_english/scripts/train_e0_ready.py').read_text()
        self.assertLess(text.index('bind_preflight(args.preflight_report'),text.index('optimizer.step()'))
        legacy=Path('research/industrial_defect_english/scripts/train_e0.py').read_text()
        self.assertLess(legacy.index('Legacy runner is diagnostic-only'),legacy.index('optimizer.step()'))

if __name__=='__main__':unittest.main()

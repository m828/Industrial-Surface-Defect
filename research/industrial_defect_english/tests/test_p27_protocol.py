"""P2.7 candidate contracts on synthetic tensors/files; no GPU or training."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import sys
from ..models.losses import joint_loss_with_protocol,loss_component_names
from ..scripts.handoff_common import execution_protocol_errors,verify_partition_identity,completion
from ..scripts.e0_contract import require_finite_loss
from ..scripts.server_preflight import verify_d4_against_source
from ..scripts.freeze_neu_development import freeze,QUARANTINE_POLICY
from .test_p27_quarantine import fixture,rows


def approved_values_for_fixture():
    cfg=json.loads(Path('research/industrial_defect_english/configs/e0_handoff_template.json').read_text())
    # Only input to isolated pure-function tests; not a real author approval.
    cfg.update(loss_protocol_id='neu_a2ms_mask_aware_candidate_v1',rng_policy='isolated_v1')
    return cfg


class P27ProtocolTests(unittest.TestCase):
    def test_candidate_settings_and_no_implicit_loss(self):
        cfg=approved_values_for_fixture();self.assertEqual(execution_protocol_errors(cfg),[])
        for key in ('loss_protocol_id','rng_policy','e0_protocol_id'):
            for value in (None,'wrong'):
                bad=copy.deepcopy(cfg);bad[key]=value
                with self.subTest(key=key,value=value):self.assertTrue(execution_protocol_errors(bad))
        for key in ('performance_early_stop','response_early_stop'):
            bad=copy.deepcopy(cfg);bad['stopping'][key]=True
            self.assertTrue(execution_protocol_errors(bad))
        self.assertFalse(cfg['stopping']['author_approved'])
        self.assertEqual(cfg['data_governance_policy_id'],'disabled')

    def test_three_conditions_use_one_explicit_loss(self):
        cfg=approved_values_for_fixture();torch.manual_seed(4)
        labels=torch.randint(0,4,(2,8,8))
        outputs=[torch.randn(2,4,8,8) for _ in range(3)]+[torch.randn(2,1,8,8)]
        losses=[]
        for variant in cfg['matrix']:
            with self.subTest(variant=variant):
                total,parts=joint_loss_with_protocol(outputs,labels,cfg['loss_weights'],cfg['loss_protocol_id'])
                self.assertEqual(len(parts),7);losses.append(total)
        self.assertTrue(all(torch.equal(losses[0],x) for x in losses))
        with self.assertRaises((ValueError,TypeError)):
            joint_loss_with_protocol(outputs,labels,cfg['loss_weights'],None)

    def test_empty_hard_pixels_write_event_and_stop(self):
        labels=torch.zeros((1,8,8),dtype=torch.long)
        confident=torch.full((1,4,8,8),-100.);confident[:,0]=100.
        outputs=[confident.clone() for _ in range(3)]+[torch.zeros(1,1,8,8)]
        protocol='neu_a2ms_mask_aware_candidate_v1'
        total,parts=joint_loss_with_protocol(outputs,labels,[10,1,3,3],protocol)
        self.assertFalse(torch.isfinite(total))
        with tempfile.TemporaryDirectory(prefix='loss event path ') as td:
            with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('NO TRAINING')) as step:
                with self.assertRaises(FloatingPointError):
                    require_finite_loss(total,parts,loss_component_names(protocol),td,6,5,protocol)
                step.assert_not_called()
            event=json.loads(Path(td,'failure_event.json').read_text())
            self.assertEqual(event['completed_updates'],5);self.assertEqual(event['attempted_update'],6)
            self.assertEqual(event['result_status'],'INCOMPLETE');self.assertEqual(event['loss_protocol_id'],protocol)
            self.assertIsNone(event['components'][0]['value'])
            self.assertEqual(completion('failed',5,10000),'INCOMPLETE')

    def test_finite_loss_has_no_failure_side_effect(self):
        with tempfile.TemporaryDirectory() as td:
            require_finite_loss(torch.tensor(1.),[torch.tensor(1.)],['component'],td,1,0,'synthetic')
            self.assertEqual(list(Path(td).iterdir()),[])

    def test_no_performance_or_response_stop_calls_in_runner(self):
        text=Path('research/industrial_defect_english/scripts/train_e0_ready.py').read_text()
        calls=[n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id=='relative_stop_flags' for n in calls))
        self.assertNotIn('collapse_count',text);self.assertNotIn('harm_count',text)
        self.assertLess(text.index('require_finite_loss(total'),text.index('optimizer.step()'))
        self.assertLess(text.index('bind_preflight(args.preflight_report'),text.index('optimizer.step()'))

    def test_quarantine_partition_and_server_recheck(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,te,parent,_=fixture(root)
            audit=freeze(tr,te,root,root/'out',expected_train=9,expected_test=1,target=3,
                         family_map=parent,quarantine_policy_id=QUARANTINE_POLICY)
            a=rows(root/'out/dev_train.jsonl');b=rows(root/'out/dev_validation.jsonl');q=rows(root/'out/quarantine.jsonl')
            pairs={tuple(line.split()) for line in tr.read_text().splitlines()}
            result=verify_partition_identity(a,b,q,audit,pairs)
            self.assertEqual(result['original_train_count'],9)
            ev=json.loads((root/'out/d4_matches.json').read_text())
            result=verify_d4_against_source(tr,te,root,a,b,ev,q,audit,QUARANTINE_POLICY)
            self.assertEqual(result['quarantined_train_n'],3)
            for bad_q in (q[:-1],q+[dict(a[0])]):
                with self.assertRaises(ValueError):verify_partition_identity(a,b,bad_q,audit,pairs)
            bad_audit=dict(audit,eligible_train_count=99)
            with self.assertRaises(ValueError):verify_partition_identity(a,b,q,bad_audit,pairs)
            bad_q=copy.deepcopy(q);bad_q[0]['quarantine_reason']='invented'
            with self.assertRaisesRegex(ValueError,'quarantine evidence'):
                verify_d4_against_source(tr,te,root,a,b,ev,bad_q,audit,QUARANTINE_POLICY)
            parent.write_text(parent.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'Parent map source/SHA'):
                verify_d4_against_source(tr,te,root,a,b,ev,q,audit,QUARANTINE_POLICY)

    def test_endpoint_comparison_rejects_different_loss(self):
        from ..scripts.compare_e0_endpoints import compare
        cfg=approved_values_for_fixture()
        base={'config':cfg,'source_commit_sha':'a'*40,'config_sha256':'b'*64}
        other=copy.deepcopy(base);other['config']['loss_protocol_id']='e0_multiclass_detail_v1'
        with patch('research.industrial_defect_english.scripts.compare_e0_endpoints.read_endpoint',
                   side_effect=[(base,{}),(other,{})]),self.assertRaisesRegex(ValueError,'execution protocol'):
            compare('SYNTHETIC_A','SYNTHETIC_B')

    def test_valid_candidate_still_rejects_missing_preflight_before_step(self):
        from ..scripts import train_e0_ready
        cfg=approved_values_for_fixture();cfg.update(scientific_gate='AUTHOR_APPROVED_E0',expected_source_sha256='synthetic')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'config.json';p.write_text(json.dumps(cfg))
            argv=['runner','--config',str(p),'--variant','formal_a2ms','--source-commit','a'*40,
                  '--run-id','fixture:formal_a2ms:s1337','--authorize-gpu']
            with patch.object(sys,'argv',argv),patch.object(train_e0_ready,'source_digest',return_value='synthetic'),\
                 patch('research.industrial_defect_english.scripts.server_preflight.preflight',return_value={'status':'BLOCKED'}) as pf,\
                 patch.object(torch.optim.Adam,'step',side_effect=AssertionError('NO TRAINING')) as step:
                with self.assertRaises(SystemExit):train_e0_ready.main()
                pf.assert_called_once();step.assert_not_called()

if __name__=='__main__':unittest.main()

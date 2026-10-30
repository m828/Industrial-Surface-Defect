"""Synthetic helper tests only. No real dataset, official checkpoint or GPU."""
import hashlib,json,tempfile,unittest,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ..scripts.freeze_neu_development import freeze
from ..scripts.e0_contract import require_finite,check_budget,response_stats,summarize_validation


class PrelaunchChecks(unittest.TestCase):
    def test_development_grouping_and_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); tr=[];te=[]; (root/'images').mkdir(); (root/'masks').mkdir()
            for i in range(10):
                # Images 0 and1 share source pixels but have distinct sample IDs.
                pixels=np.full((12,12,3),max(i,1),dtype=np.uint8)
                im=root/'images'/f'i{i}.png';gt=root/'masks'/f'i{i}.png'
                Image.fromarray(pixels).save(im)
                Image.fromarray(np.tile(np.arange(12,dtype=np.uint8)%4,(12,1))).save(gt)
                (tr if i<8 else te).append(f'masks/{gt.name} images/{im.name}\n')
            train=root/'official_train.txt';train.write_text(''.join(tr))
            test=root/'official_test.txt';test.write_text(''.join(te))
            audit=freeze(train,test,root,root/'dev',target=2,expected_train=8,expected_test=2)
            self.assertEqual(audit['actual_train']+audit['actual_validation'],8)
            self.assertEqual(audit['status'],'CANDIDATE_FAMILY_LINEAGE_UNCONFIRMED')
            read=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
            a=read(root/'dev/dev_train.jsonl');b=read(root/'dev/dev_validation.jsonl')
            self.assertFalse({r['group_id'] for r in a}&{r['group_id'] for r in b})
            # Corrupting the official test identity must stop, never silently drop a row.
            test.write_text(tr[0]+te[1])
            with self.assertRaisesRegex(ValueError,'Cross-original-split'):freeze(train,test,root,root/'bad',target=2,expected_train=8,expected_test=2)

    def test_shape_or_label_mismatch_rejected(self):
        from ..scripts.freeze_neu_development import scan
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'masks').mkdir(); Image.fromarray(np.zeros((9,9,3),dtype=np.uint8)).save(root/'i.png')
            Image.fromarray(np.zeros((8,9),dtype=np.uint8)).save(root/'masks/i.png')
            p=root/'list';p.write_text('masks/i.png i.png\n')
            with self.assertRaisesRegex(ValueError,'shape'):scan(p,root,'train')

    def test_nonfinite_and_budget_guards(self):
        with self.assertRaises(FloatingPointError):require_finite([('named_grad',torch.tensor(float('nan')))],'gradient')
        require_finite([('unused',None),('valid',torch.ones(2))],'gradient')
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError,'wall'):check_budget(time.perf_counter()-10,td,{'max_wall_seconds_per_run':1,'max_run_bytes':10})
            Path(td,'x').write_bytes(b'1234')
            with self.assertRaisesRegex(RuntimeError,'disk'):check_budget(time.perf_counter(),td,{'max_wall_seconds_per_run':100,'max_run_bytes':1})

    def test_relative_response(self):
        l=torch.ones(1,3,5,5);r=3*l;t=.2*l;b=.1*torch.ones(1,1,5,5)
        result=response_stats(l,r,t,b)
        self.assertAlmostEqual(result['rho_difference'],.1,places=5)
        self.assertAlmostEqual(result['rho_detail'],.2,places=5)
        self.assertAlmostEqual(result['beta_spatial_std'],0.)

    def test_confusion_error_types(self):
        summary=summarize_validation({'confusion':[[4,1,0],[1,2,1],[0,2,3]],'cases':[]})
        self.assertEqual(summary['background_fp_pixels'],1)
        self.assertEqual(summary['foreground_cross_class_pixels'],3)
        self.assertEqual(summary['foreground_to_background_pixels'],1)

    def test_relative_stop_contract(self):
        from ..scripts.e0_contract import relative_stop_flags
        baseline={'miou_including_background':.9,'iou':[.95,.88,.87],
                  'confusion':[[90,10,0],[1,80,2],[1,2,80]],'cases':[]}
        candidate=dict(baseline,miou_including_background=.87,iou=[.95,.80,.87],
                       confusion=[[85,15,0],[1,80,2],[1,2,80]])
        flags=relative_stop_flags(candidate,baseline,{'approved_miou_drop':.01,
              'approved_class_iou_tolerances':{'1':.02,'2':.02},'approved_background_fp_relative_increase':.1})
        self.assertIn('overall_miou_harm',flags);self.assertIn('class_1_harm',flags)
        self.assertIn('background_fp_harm',flags)

    def test_ready_runner_blocks_missing_assets(self):
        import sys,subprocess
        from ..scripts.train_e0 import source_digest
        with tempfile.TemporaryDirectory() as td:
            cfg=json.loads(Path('research/industrial_defect_english/configs/e0_neu_prelaunch.json').read_text())
            cfg['expected_source_sha256']=source_digest()
            p=Path(td)/'config.json';p.write_text(json.dumps(cfg))
            result=subprocess.run([sys.executable,'-m','research.industrial_defect_english.scripts.train_e0_ready',
                '--config',str(p),'--variant','coupled_transport','--preflight-only'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Missing or mismatched split manifest SHA',result.stderr)

    def test_weights_helper_synthetic_only(self):
        from ..models import build_model
        from ..scripts.verify_formal_checkpoint import verify
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'SYNTHETIC_NOT_OFFICIAL.pt'
            torch.save({'model_state':build_model('formal_a2ms').state_dict()},p)
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            result=verify(p,h)
            self.assertEqual(result['formal_a2ms_strict'],'PASS')
            self.assertEqual(len(result['coupled_transport']['missing']),4)
            with self.assertRaisesRegex(ValueError,'SHA'):verify(p,'wrong')

if __name__=='__main__':unittest.main()

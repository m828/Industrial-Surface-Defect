import unittest,subprocess,sys
from pathlib import Path
import torch
from ..models.metrics import case_metrics,boundary_scores
from ..models import build_model


class UtilityChecks(unittest.TestCase):
    def test_empty_and_perfect_boundaries(self):
        z=torch.zeros(15,17,dtype=torch.bool); self.assertEqual(boundary_scores(z,z),(None,None))
        z[4:10,5:11]=True
        self.assertEqual(boundary_scores(z,z),(1.,1.))
        self.assertEqual(boundary_scores(z,torch.zeros_like(z)),(0.,0.))
    def test_error_types(self):
        y=torch.tensor([[0,1,2],[0,1,2]]); p=torch.tensor([[1,2,2],[0,1,1]])
        r=case_metrics(y,p,3,'synthetic')[0]
        self.assertEqual((r['tp'],r['fp_background'],r['fp_other_class'],r['fn']),(1,1,1,1))
    def test_gpu_guard(self):
        cmd=[sys.executable,'-m','research.industrial_defect_english.scripts.train_e0','--config',
             'research/industrial_defect_english/configs/e0_neu.json','--variant','coupled_transport','--preflight-only']
        p=subprocess.run(cmd,capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0); self.assertIn('scientific/author gate remains pending',p.stderr)
    def test_synthetic_dataset_adapter_and_sha(self):
        import tempfile,json,numpy as np,hashlib
        from PIL import Image
        from ..scripts.train_e0 import records,SegmentationData
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); im=root/'image.png'; gt=root/'mask.png'
            Image.fromarray(np.zeros((25,27,3),dtype=np.uint8)).save(im)
            labels=np.zeros((25,27),dtype=np.uint8); labels[4:11,5:13]=2
            Image.fromarray(labels).save(gt)
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            row={'sample_id':'synthetic','original_split':'train','image':str(im),'mask':str(gt),
                 'image_sha256':sha(im),'mask_sha256':sha(gt)}
            manifest=root/'train.jsonl'; manifest.write_text(json.dumps(row)+'\n')
            rows=records(str(manifest),sha(manifest)); x,y=SegmentationData(rows,[200,200])[0]
            self.assertEqual(list(x.shape),[3,200,200]); self.assertEqual(set(y.unique().tolist()),{0,2})
            with self.assertRaises(ValueError): records(str(manifest),'wrong_sha')

    def test_fixed_validation_class_set(self):
        from ..scripts.train_e0 import validate
        from torch.utils.data import Dataset,DataLoader
        from unittest.mock import patch
        class Toy(Dataset):
            rows=[{'sample_id':'synthetic'}]
            def __len__(self): return 1
            def __getitem__(self,i):
                y=torch.tensor([[0,1],[2,3]])
                x=torch.nn.functional.one_hot(y,4).permute(2,0,1).float()*10
                return x,y
        class Echo(torch.nn.Module):
            def forward(self,x): return [x]
        with patch.object(torch.Tensor,'cuda',lambda self:self):
            result=validate(Echo(),DataLoader(Toy(),batch_size=1),4)
        self.assertEqual(result['miou_including_background'],1.)
        self.assertEqual(result['declared_class_ids'],[0,1,2,3])

    def test_all_ablation_shapes(self):
        torch.set_num_threads(4)
        for v in ['stage1_only','no_head_shift','head_shift_only','uncoupled_transport','recomputed_final_transport']:
            model=build_model(v).eval()
            with torch.no_grad(): out=model(torch.randn(1,3,200,200))
            self.assertEqual(list(out[0].shape),[1,4,200,200])

if __name__=='__main__': unittest.main()

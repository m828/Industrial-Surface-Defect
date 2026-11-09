"""KSDD2 task-adapter checks using synthetic inputs/metadata only."""
import copy,hashlib,json,random,unittest
import numpy as np
import torch
from ..models import build_model
from ..models.losses import joint_loss_with_protocol,LOSS_PROTOCOL_HISTORICAL,_historical_ohem,_historical_bce,_historical_detail_loss
from ..scripts.ksdd2_support import (MATRIX,BINARY_LOSS_ID,decode_mask,pad_native,
  masked_ohem,binary_joint_loss,binary_metrics,development_split,area_threshold,initialization_audit)


def outputs(n=2,h=32,w=48):
    return [torch.randn(n,2,h,w,requires_grad=True) for _ in range(3)]+[torch.randn(n,1,h,w,requires_grad=True)]


class KSDD2Tests(unittest.TestCase):
    def test_four_models_two_logits_shared_head_and_training_detail(self):
        for variant in MATRIX:
            model=build_model(variant,num_classes=2,seed=1337).train();seen=[];detail=[]
            a=model.seg_heads.register_forward_hook(lambda m,x,y:seen.append((id(m),tuple(y.shape))))
            b=model.seg_head_detailloss.register_forward_hook(lambda m,x,y:detail.append(tuple(y.shape)))
            with torch.no_grad():out=model(torch.randn(2,3,224,256))
            self.assertEqual([tuple(x.shape) for x in out],[(2,2,224,256)]*3+[(2,1,224,256)])
            self.assertEqual(len(seen),3);self.assertEqual(len({i for i,_ in seen}),1);self.assertEqual(len(detail),1)
            self.assertTrue(all(torch.isfinite(x).all() for x in out));seen.clear();detail.clear();model.eval()
            with torch.no_grad():out=model(torch.randn(1,3,224,256))
            self.assertEqual(len(out),1);self.assertEqual(tuple(out[0].shape),(1,2,224,256));self.assertFalse(detail)
            a.remove();b.remove()

    def test_binary_common_initialization_and_transport_parameters(self):
        for seed in (1337,2027,3407):
            result=initialization_audit(seed);self.assertEqual(result['status'],'PASS');self.assertEqual(result['num_classes'],2)
            self.assertEqual(len({x['common_tensor_collection_sha256'] for x in result['checks'].values()}),1)
            hashes={v['transport_initial_tensor_sha256'] for k,v in result['checks'].items() if k!='formal_a2ms'}
            self.assertEqual(len(hashes),1)

    def test_raw_mask_encoding_is_explicit_and_not_ignore255(self):
        raw=np.array([[0,255],[255,0]],np.uint8)
        self.assertEqual(decode_mask(raw,'zero_255').tolist(),[[0,1],[1,0]])
        with self.assertRaises(ValueError):decode_mask(raw,'zero_one')
        with self.assertRaises(ValueError):decode_mask(np.zeros((2,2,3),np.uint8),'zero_255')
        with self.assertRaises(ValueError):decode_mask(np.array([[2]],np.uint8),'zero_255')

    def test_native_padding_preserves_single_pixels_and_shape(self):
        image=np.zeros((23,41,3),np.uint8);gt=np.zeros((23,41),np.int64);gt[5,39]=1
        x,y,v=pad_native(image,gt,[224,256]);self.assertEqual(tuple(x.shape),(3,224,256))
        self.assertEqual(int(y.sum()),1);self.assertEqual(int(v.sum()),23*41);self.assertEqual(int(y[5,39]),1)
        self.assertFalse(v[23:,:].any());self.assertFalse(v[:,41:].any())
        with self.assertRaises(ValueError):pad_native(image,gt,[32,32])
        with self.assertRaises(ValueError):pad_native(image,gt,[33,64])

    def test_historical_binary_mask_aware_repeats_segmentation_routes(self):
        ys=torch.zeros(2,32,48,dtype=torch.long);ys[1,12:15,12:15]=1;oo=outputs()
        total,parts=joint_loss_with_protocol(oo,ys,[10,1,3,3],LOSS_PROTOCOL_HISTORICAL)
        for i in range(3):torch.testing.assert_close(parts[i],parts[4+i],rtol=0,atol=0)
        expected=2*(10*parts[0]+parts[1]+3*parts[2])+3*parts[3]
        torch.testing.assert_close(total,expected)

    def test_empty_gt_candidate_is_finite_and_strict_ohem_edge_is_recorded(self):
        labels=torch.zeros(2,32,48,dtype=torch.long);valid=torch.ones_like(labels,dtype=torch.bool)
        logits=torch.stack((torch.full_like(labels,1000.,dtype=torch.float32),torch.full_like(labels,-1000.,dtype=torch.float32)),1).requires_grad_()
        self.assertFalse(torch.isfinite(_historical_ohem(logits,labels)))
        a,empty=masked_ohem(logits,labels,valid);self.assertTrue(empty);self.assertTrue(torch.isfinite(a))
        oo=[logits,logits.clone(),logits.clone(),torch.zeros(2,1,32,48,requires_grad=True)]
        loss,parts,info=binary_joint_loss(oo,labels,valid,[10,1,3,3],BINARY_LOSS_ID)
        self.assertEqual(info['empty_hard_pixel_ce_fallback_routes'],[0,2]);self.assertTrue(torch.isfinite(loss))
        loss.backward();self.assertTrue(torch.isfinite(oo[3].grad).all())

    def test_candidate_tiny_gt_padding_zero_gradient_and_explicit_protocol(self):
        labels=torch.zeros(2,32,48,dtype=torch.long);labels[1,14,14]=1
        valid=torch.zeros_like(labels,dtype=torch.bool);valid[:,:24,:40]=True;oo=outputs()
        loss,parts,info=binary_joint_loss(oo,labels,valid,[10,1,3,3],BINARY_LOSS_ID)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(len(parts),4);loss.backward()
        for tensor in oo:
            self.assertIsNotNone(tensor.grad);self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertEqual(float(tensor.grad[:,:,24:,:].abs().sum()),0.)
            self.assertEqual(float(tensor.grad[:,:,:,40:].abs().sum()),0.)
        with self.assertRaises(ValueError):binary_joint_loss(oo,labels,valid,[10,1,3,3],None)
        with self.assertRaises(ValueError):binary_joint_loss(oo,labels,torch.zeros_like(valid),[10,1,3,3],BINARY_LOSS_ID)

    def test_metrics_four_empty_cases_and_normal_false_positives(self):
        z=np.zeros((8,8),np.int64);one=z.copy();one[3,4]=1
        result=binary_metrics([('tp',one,one),('fn',one,z),('fp',z,one),('empty',z,z)])
        self.assertEqual(result['tp'],1);self.assertEqual(result['fp'],1);self.assertEqual(result['fn'],1)
        self.assertAlmostEqual(result['foreground_iou'],1/3);self.assertAlmostEqual(result['foreground_dice'],.5)
        self.assertAlmostEqual(result['gt_positive_macro_iou'],.5);self.assertEqual(result['case_macro_iou_n'],3)
        self.assertEqual(result['both_empty_n'],1);self.assertEqual(result['empty_gt_n'],2)
        self.assertEqual(result['empty_gt_image_fpr'],.5);self.assertIsNone(result['cases'][-1]['iou'])
        all_empty=binary_metrics([('normal',z,z)]);self.assertIsNone(all_empty['foreground_iou'])
        self.assertIsNone(all_empty['case_macro_foreground_iou']);self.assertEqual(all_empty['background_iou'],1.)

    def test_small_groups_and_aggregate_vs_case_macro_have_distinct_denominators(self):
        a=np.ones((8,8),np.int64);b=np.zeros_like(a);b[0,0]=1
        result=binary_metrics([('large',a,a),('small',b,np.zeros_like(b))],small_area_threshold=.05)
        self.assertEqual(result['gt_positive_macro_iou'],.5);self.assertAlmostEqual(result['foreground_iou'],64/65)
        self.assertEqual(result['small_positive_n'],1);self.assertEqual(result['small_positive_macro_dice'],0.)

    def records(self):
        return [{'sample_id':f'train/{i:03d}','original_split':'train','gt_positive':i<10,
                 'gt_fraction':(.01+i*.001) if i<10 else 0.,
                 'image_pixel_sha256':hashlib.sha256(str(i).encode()).hexdigest(),'parent_id':None} for i in range(40)]

    def test_grouped_split_is_deterministic_and_official_test_excluded(self):
        rows=self.records();rows[1]['parent_id']=rows[0]['parent_id']='partA'
        rows[3]['image_d4_canonical_sha256']=rows[2]['image_d4_canonical_sha256']='f'*64
        ids={r['sample_id'] for r in rows}
        a,b,info=development_split(rows,ids,{'test/000'})
        aa,bb,ii=development_split(list(reversed(rows)),ids,{'test/000'})
        self.assertEqual((a,b,info),(aa,bb,ii));self.assertEqual(len(a)+len(b),40)
        self.assertFalse({r['group_id'] for r in a}&{r['group_id'] for r in b})
        self.assertEqual(info['parent_metadata_status'],'UNKNOWN_OR_PARTIAL')
        self.assertEqual(rows[0].get('group_id'),None)
        with self.assertRaises(ValueError):development_split(rows,ids,{'train/000'})
        bad=copy.deepcopy(rows);bad[0]['original_split']='test'
        with self.assertRaises(ValueError):development_split(bad,ids,set())

    def test_group_coverage_failure_unknown_parent_and_training_only_area_threshold(self):
        rows=self.records();ids={r['sample_id'] for r in rows}
        for row in rows:row['parent_id']='same_capture'
        with self.assertRaises(ValueError):development_split(rows,ids,set())
        rows=self.records();rows[0]['parent_id']='UNKNOWN'
        with self.assertRaises(ValueError):development_split(rows,ids,set())
        cut=area_threshold(self.records());self.assertEqual(cut['n'],10)
        self.assertAlmostEqual(cut['small_area_fraction_p33'],.013)
        with self.assertRaises(ValueError):area_threshold([r for r in self.records() if not r['gt_positive']])

if __name__=='__main__':unittest.main()

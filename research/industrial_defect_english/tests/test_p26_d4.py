"""Synthetic D4 split-protection tests; never reads real NEU data."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from ..scripts.freeze_neu_development import d4_match, d4_transforms, freeze, scan


def write_pair(root, sample, image, mask):
    for directory in ('images','masks'):(root/directory).mkdir(exist_ok=True)
    Image.fromarray(np.asarray(image,dtype=np.uint8)).save(root/'images'/f'{sample}.png')
    Image.fromarray(np.asarray(mask,dtype=np.uint8)).save(root/'masks'/f'{sample}.png')
    return f'masks/{sample}.png images/{sample}.png\n'


class D4SplitTests(unittest.TestCase):
    def test_all_eight_square_and_nonsquare(self):
        for shape in ((4,4,3),(3,5,3)):
            image=np.arange(np.prod(shape),dtype=np.uint8).reshape(shape)
            variants=dict(d4_transforms(image))
            self.assertEqual(len(variants),8)
            self.assertEqual(len({(x.shape,x.tobytes()) for x in variants.values()}),8)
            np.testing.assert_array_equal(variants['r180'],np.rot90(image,2))

    def test_all_eight_transform_matches_are_verified_against_decoded_images(self):
        with tempfile.TemporaryDirectory(prefix='d4 all ') as td:
            root=Path(td);image=np.arange(3*5*3,dtype=np.uint8).reshape(3,5,3)
            mask=np.arange(15,dtype=np.uint8).reshape(3,5)%4
            lines=[write_pair(root,'source',image,mask)]
            for name,im in d4_transforms(image):
                gt=dict(d4_transforms(mask))[name]
                lines.append(write_pair(root,name,im,gt))
            (root/'pairs').write_text(''.join(lines));rows=scan(root/'pairs',root,'train')
            for name,row in zip(dict(d4_transforms(image)),rows[1:]):
                match=d4_match(rows[0],row)
                self.assertEqual([m['transform_first_to_second'] for m in match],[name])
                self.assertTrue(match[0]['gt_equal'])
                self.assertEqual(rows[0]['image_d4_canonical_sha256'],row['image_d4_canonical_sha256'])

    def test_known_shape_cross_split_blocked_and_gt_difference_preserved(self):
        with tempfile.TemporaryDirectory(prefix='d4 test ') as td:
            root=Path(td);image=np.arange(3*5*3,dtype=np.uint8).reshape(3,5,3)
            mask=np.arange(15,dtype=np.uint8).reshape(3,5)%4
            train=write_pair(root,'004097',image,mask)
            rotated=np.rot90(mask,2).copy();rotated[0,0]=(rotated[0,0]+1)%4
            test=write_pair(root,'000717',np.rot90(image,2),rotated)
            (root/'train list').write_text(train);(root/'test list').write_text(test)
            out=root/'output with spaces'
            args=(root/'train list',root/'test list',root,out)
            with self.assertRaisesRegex(ValueError,'Cross-original-split'):
                freeze(*args,expected_train=1,expected_test=1,target=1)
            evidence=json.loads((out/'d4_matches.json').read_text());p=evidence['train_test_pairs'][0]
            self.assertEqual((p['first_sample_id'],p['second_sample_id']),('004097','000717'))
            r180=next(m for m in p['matches'] if m['transform_first_to_second']=='r180')
            self.assertEqual(r180['gt_mismatch_pixels'],1);self.assertFalse(r180['gt_equal'])
            self.assertEqual(json.loads((out/'split_audit.json').read_text())['status'],'BLOCKED_CROSS_SPLIT_D4')
            self.assertEqual(json.loads((out/'split_audit.json').read_text())['d4_exact_check_status'],'BLOCKED_CROSS_SPLIT_D4')
            self.assertFalse((out/'dev_train.jsonl').exists());self.assertFalse((out/'dev_validation.jsonl').exists())
            saved=(out/'d4_matches.json').read_bytes()
            with self.assertRaises(FileExistsError):freeze(*args,expected_train=1,expected_test=1,target=1)
            self.assertEqual(saved,(out/'d4_matches.json').read_bytes())

    def test_training_groups_dont_cross_and_manifest_is_deterministic(self):
        with tempfile.TemporaryDirectory(prefix='d4 groups ') as td:
            root=Path(td);mask=np.arange(16,dtype=np.uint8).reshape(4,4)%4
            first=np.arange(48,dtype=np.uint8).reshape(4,4,3)
            train=[write_pair(root,'a',first,mask),write_pair(root,'b',np.flip(first,1),np.flip(mask,1))]
            for i in range(5):
                train.append(write_pair(root,f'x{i}',np.full((4,4,3),100+i,dtype=np.uint8),mask))
            test=[write_pair(root,'z',np.full((4,4,3),210,dtype=np.uint8),mask)]
            tr=root/'train.txt';te=root/'test.txt';tr.write_text(''.join(train));te.write_text(''.join(test))
            a=freeze(tr,te,root,root/'output one',target=2,expected_train=7,expected_test=1)
            b=freeze(tr,te,root,root/'output two',target=2,expected_train=7,expected_test=1)
            self.assertEqual(a['lineage_evidence'],'UNKNOWN')
            self.assertEqual(a['status'],'CANDIDATE_FAMILY_LINEAGE_UNCONFIRMED')
            self.assertEqual(a['d4_exact_check_status'],'PASS_NO_CROSS_SPLIT_EXACT_D4')
            self.assertEqual(a['d4_transform_set'],'D4_EXACT_RGB_8_V1')
            self.assertEqual(a['d4_train_test_pair_count'],0)
            self.assertEqual(a['dev_train_sha256'],b['dev_train_sha256'])
            self.assertEqual(a['dev_validation_sha256'],b['dev_validation_sha256'])
            rows=lambda path:[json.loads(x) for x in path.read_text().splitlines()]
            left=rows(root/'output one/dev_train.jsonl');right=rows(root/'output one/dev_validation.jsonl')
            self.assertFalse({r['group_id'] for r in left}&{r['group_id'] for r in right})
            for subset in (left,right):
                ids={r['sample_id'] for r in subset};self.assertEqual('a' in ids,'b' in ids)
            evidence=json.loads((root/'output one/d4_matches.json').read_text())
            self.assertTrue(any(p['first_sample_id']=='a' and p['second_sample_id']=='b' for p in evidence['train_train_pairs']))

    def test_invalid_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);line=write_pair(root,'a',np.zeros((4,4,3)),np.full((4,4),4))
            (root/'train.txt').write_text(line)
            with self.assertRaisesRegex(ValueError,'Invalid mask values'):scan(root/'train.txt',root,'train')

if __name__=='__main__':unittest.main()

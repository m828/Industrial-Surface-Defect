"""Candidate quarantine isolation on synthetic PNGs only; no real NEU scan."""
import csv,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from ..scripts.freeze_neu_development import QUARANTINE_POLICY,freeze,scan,d4_evidence,partition_training


def fixture(root,transform='r180',cross=True):
    (root/'images').mkdir();(root/'masks').mkdir()
    first=np.arange(48,dtype=np.uint8).reshape(4,4,3)
    mask=np.arange(16,dtype=np.uint8).reshape(4,4)%4
    images={'a':first,'b':np.flip(first,1),'c':np.full((4,4,3),88,dtype=np.uint8)}
    images.update({f'x{i}':np.full((4,4,3),110+i,dtype=np.uint8) for i in range(6)})
    train_ids=list(images)
    test=np.rot90(first,2) if transform=='r180' else np.flip(first,1)
    images['test']=test if cross else np.full((4,4,3),230,dtype=np.uint8)
    for sid,image in images.items():
        Image.fromarray(image).save(root/'images'/f'{sid}.png')
        # Keep GT distinct from any attempted reasoning about matching image identity.
        gt=mask.copy()
        if sid=='test':gt[0,0]=3
        Image.fromarray(gt).save(root/'masks'/f'{sid}.png')
    line=lambda sid:f'masks/{sid}.png images/{sid}.png\n'
    tr=root/'train.txt';te=root/'test.txt'
    tr.write_text(''.join(line(sid) for sid in train_ids));te.write_text(line('test'))
    parent=root/'parent.csv'
    with parent.open('w') as f:
        writer=csv.writer(f);writer.writerow(['image_relative','parent_id'])
        writer.writerows((f'images/{sid}.png','same_family' if sid in ('a','b','c') else sid) for sid in train_ids)
    return tr,te,parent,train_ids


def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class QuarantineTests(unittest.TestCase):
    def test_default_off_stops_cross_split(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,te,_,_=fixture(root)
            with self.assertRaisesRegex(ValueError,'Cross-original-split'):
                freeze(tr,te,root,root/'out',expected_train=9,expected_test=1,target=3)
            self.assertFalse((root/'out/dev_train.jsonl').exists())

    def test_full_connected_group_quarantined_and_count_identity(self):
        with tempfile.TemporaryDirectory(prefix='quarantine path ') as td:
            root=Path(td);tr,te,family,original_ids=fixture(root)
            before={p:p.read_bytes() for p in root.rglob('*.png')}
            audit=freeze(tr,te,root,root/'out',expected_train=9,expected_test=1,target=3,
                         family_map=family,quarantine_policy_id=QUARANTINE_POLICY)
            a=rows(root/'out/dev_train.jsonl');b=rows(root/'out/dev_validation.jsonl');q=rows(root/'out/quarantine.jsonl')
            a_ids={r['sample_id'] for r in a};b_ids={r['sample_id'] for r in b};q_ids={r['sample_id'] for r in q}
            self.assertEqual(q_ids,{'a','b','c'});self.assertEqual(a_ids|b_ids|q_ids,set(original_ids))
            self.assertFalse(a_ids&b_ids or (a_ids|b_ids)&q_ids)
            self.assertEqual(audit['original_train_count'],audit['eligible_train_count']+audit['quarantined_train_count'])
            self.assertEqual(audit['actual_train']+audit['actual_validation'],audit['eligible_train_count'])
            self.assertEqual(audit['quarantine_manifest_sha256'],hashlib.sha256((root/'out/quarantine.jsonl').read_bytes()).hexdigest())
            self.assertEqual(audit['d4_exact_check_status'],'PASS_CROSS_SPLIT_EXACT_D4_QUARANTINED')
            self.assertEqual(audit['data_governance_policy_id'],QUARANTINE_POLICY)
            propagated=next(r for r in q if r['sample_id']=='c')
            self.assertEqual(propagated['quarantine_reason'],'TRAIN_GROUP_CONNECTED_TO_EXACT_D4_TEST_MATCH')
            self.assertTrue(propagated['quarantine_matches'])
            self.assertEqual(propagated['d4_evidence_sha256'],audit['d4_matches_sha256'])
            self.assertTrue(all(p.read_bytes()==v for p,v in before.items()))

    def test_mirror_quarantine_keeps_unknown_parent_status(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,te,_,_=fixture(root,transform='flip_h')
            audit=freeze(tr,te,root,root/'out',expected_train=9,expected_test=1,target=3,
                         quarantine_policy_id=QUARANTINE_POLICY,no_derivatives_attested=True)
            self.assertEqual({r['sample_id'] for r in rows(root/'out/quarantine.jsonl')},{'a','b'})
            self.assertEqual(audit['status'],'CANDIDATE_FAMILY_LINEAGE_UNCONFIRMED')
            self.assertNotEqual(audit['lineage_evidence'],'SUPPLIED_PARENT_MAP')

    def test_empty_quarantine_is_written_and_output_protected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,te,family,_=fixture(root,cross=False)
            args=(tr,te,root,root/'out')
            kw=dict(expected_train=9,expected_test=1,target=3,family_map=family,quarantine_policy_id=QUARANTINE_POLICY)
            audit=freeze(*args,**kw)
            self.assertEqual((root/'out/quarantine.jsonl').read_bytes(),b'')
            self.assertEqual(audit['eligible_train_count'],9);self.assertEqual(audit['quarantined_train_count'],0)
            with self.assertRaises(FileExistsError):freeze(*args,**kw)

    def test_parent_conflict_and_partial_map_block(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,te,family,_=fixture(root)
            original=family.read_text();family.write_text(original.replace('images/b.png,same_family','images/b.png,different'))
            with self.assertRaisesRegex(ValueError,'parent IDs conflict'):
                freeze(tr,te,root,root/'conflict',expected_train=9,expected_test=1,target=3,
                       family_map=family,quarantine_policy_id=QUARANTINE_POLICY)
            self.assertFalse((root/'conflict').exists())
            family.write_text('\n'.join(original.splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError,'cover all'):
                freeze(tr,te,root,root/'partial',expected_train=9,expected_test=1,target=3,
                       family_map=family,quarantine_policy_id=QUARANTINE_POLICY)

    def test_deterministic_groups_and_unknown_policy_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,te,family,_=fixture(root,cross=False)
            a=freeze(tr,te,root,root/'one',expected_train=9,expected_test=1,target=3,family_map=family,quarantine_policy_id=QUARANTINE_POLICY)
            b=freeze(tr,te,root,root/'two',expected_train=9,expected_test=1,target=3,family_map=family,quarantine_policy_id=QUARANTINE_POLICY)
            self.assertEqual(a['dev_train_sha256'],b['dev_train_sha256']);self.assertEqual(a['dev_validation_sha256'],b['dev_validation_sha256'])
            first=rows(root/'one/dev_train.jsonl');second=rows(root/'one/dev_validation.jsonl')
            self.assertFalse({r['group_id'] for r in first}&{r['group_id'] for r in second})
            with self.assertRaisesRegex(ValueError,'Unknown quarantine'):
                freeze(tr,te,root,root/'bad',expected_train=9,expected_test=1,target=3,quarantine_policy_id='unapproved_auto')

    def test_forged_unknown_training_id_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,te,_,_=fixture(root)
            train=scan(tr,root,'train');test=scan(te,root,'test');evidence=d4_evidence(train,test)
            evidence['train_test_pairs'][0]['first_sample_id']='unknown_train'
            with self.assertRaisesRegex(ValueError,'unknown train ID'):
                partition_training(train,evidence,quarantine_policy_id=QUARANTINE_POLICY)


if __name__=='__main__':unittest.main()

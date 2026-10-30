"""Synthetic-only tests for explicit exploratory lineage mode."""
import tempfile,unittest
from pathlib import Path
from .test_p27_quarantine import fixture,rows
from ..scripts.freeze_neu_development import freeze,QUARANTINE_POLICY
from ..scripts.handoff_common import verify_partition_identity
from ..scripts.server_preflight import verify_d4_against_source
import json

class ExploratoryD4Tests(unittest.TestCase):
    def test_default_still_blocks_parent_claim(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,te,_,_=fixture(root)
            audit=freeze(tr,te,root,root/'out',expected_train=9,expected_test=1,target=3,quarantine_policy_id=QUARANTINE_POLICY)
            self.assertEqual(audit['status'],'CANDIDATE_FAMILY_LINEAGE_UNCONFIRMED')
            self.assertEqual(audit['origin_independence'],'UNKNOWN')

    def test_authorized_exploratory_groups_and_recheck(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,te,_,_=fixture(root)
            audit=freeze(tr,te,root,root/'out',expected_train=9,expected_test=1,target=3,quarantine_policy_id=QUARANTINE_POLICY,exploratory_unknown_parent_authorized=True)
            self.assertEqual(audit['status'],'FROZEN_EXPLORATORY_D4_ONLY')
            self.assertEqual(audit['lineage_evidence'],'UNKNOWN')
            self.assertEqual(audit['origin_independence'],'UNKNOWN')
            a=rows(root/'out/dev_train.jsonl');b=rows(root/'out/dev_validation.jsonl');q=rows(root/'out/quarantine.jsonl')
            self.assertEqual({r['sample_id'] for r in q},{'a','b'})
            verify_partition_identity(a,b,q,audit,{tuple(x.split()) for x in tr.read_text().splitlines()})
            evidence=json.loads((root/'out/d4_matches.json').read_text())
            self.assertEqual(verify_d4_against_source(tr,te,root,a,b,evidence,q,audit,QUARANTINE_POLICY)['quarantined_train_n'],2)

    def test_no_quarantine_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,te,_,_=fixture(root)
            with self.assertRaises(ValueError):freeze(tr,te,root,root/'out',expected_train=9,expected_test=1,target=3,exploratory_unknown_parent_authorized=True)

if __name__=='__main__':unittest.main()

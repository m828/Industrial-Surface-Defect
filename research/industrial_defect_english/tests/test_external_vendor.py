"""Synthetic checks for external vendor byte pinning; no external assets."""
import hashlib,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from ..scripts import handoff_common as h

class ExternalVendor(unittest.TestCase):
 def test_vendor_hash_and_source_tracking(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);prefix='research/industrial_defect_english/'
   files=['scripts/handoff_common.py','scripts/server_preflight.py','scripts/train_e0_ready.py','models/factory.py','models/losses.py','models/transport.py','models/metrics.py','vendor/model_resnet.py','vendor/model_dsmo_rs50.py','vendor/model_dsmo_rs50_eSE_adapt_detailloss_fixed.py','vendor/loss.py']
   for name in files:
    p=root/prefix/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
   vendor=[{'copy':prefix+n,'sha256':hashlib.sha256(n.encode()).hexdigest()} for n in files if n.startswith('vendor/')]
   (root/prefix/'vendor/provenance.json').write_text(json.dumps(vendor))
   def run(cmd,**kw):return types.SimpleNamespace(returncode=int('/vendor/' in cmd[-1] and cmd[-1].endswith('.py')))
   def check(cmd,**kw):return (root/cmd[-1].removeprefix('HEAD:')).read_bytes()
   with patch.object(h.sys,'modules',{}),patch.object(h.subprocess,'run',side_effect=run),patch.object(h.subprocess,'check_output',side_effect=check):
    actual=h.verify_loaded_sources(root);self.assertEqual(len(actual['committed_file_sha256']),12)
    (root/prefix/'vendor/loss.py').write_text('changed')
    with self.assertRaisesRegex(ValueError,'external_vendor_sha_mismatch'):h.verify_loaded_sources(root)
 def test_manifest_cannot_add_vendor_files(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p=root/'research/industrial_defect_english/vendor/provenance.json';p.parent.mkdir(parents=True);p.write_text('[]')
   with patch.object(h.sys,'modules',{}):
    with self.assertRaisesRegex(ValueError,'vendor_lock_manifest_invalid'):h.verify_loaded_sources(root)
if __name__=='__main__':unittest.main()

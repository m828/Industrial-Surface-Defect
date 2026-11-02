"""E1-B synthetic CPU contracts; no real data, checkpoint or GPU training."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from ..models import build_model
from ..scripts.handoff_common import E1B_MATRIX,protocol_errors,mode_errors,sha256
from ..scripts.initialization_audit import audit
from ..scripts.prepare_e1b import prepare,E0_COMMIT,E1A_COMMIT
from ..scripts.summarize_e1b import summarize,read_package
from ..scripts.export_e0_artifacts import export,verify_package
from .test_handoff import fixture


def base():return json.loads(Path('research/industrial_defect_english/configs/e0_exploratory_d4_runtime_template_v1.json').read_text())


class E1BTests(unittest.TestCase):
    def test_modes_match_features_heads_and_main_output(self):
        for variant in E1B_MATRIX:
            model=build_model(variant).train()
            with torch.no_grad():model.transport.predict[-1].bias.fill_(.2)
            seen={'up':[],'lap':[],'t':[],'fusion':[],'arm':[],'head':[],'seg':[]}
            handles=[model.conv_up.register_forward_hook(lambda m,a,o:seen['up'].append(o)),
                     model.laplacian.register_forward_hook(lambda m,a,o:seen['lap'].append(o)),
                     model.transport.register_forward_hook(lambda m,a,o:seen['t'].append(o[0])),
                     model.edge_fusion.register_forward_pre_hook(lambda m,a:seen['fusion'].append(a[0])),
                     model.arm2.register_forward_pre_hook(lambda m,a:seen['arm'].append(a)),
                     model.seg_head_detailloss.register_forward_pre_hook(lambda m,a:seen['head'].append(a[0])),
                     model.seg_heads.register_forward_hook(lambda m,a,o:seen['seg'].append((id(m),o)))]
            with torch.no_grad():out=model(torch.randn(2,3,200,200))
            e,r,m=seen['up'];t=seen['t'][0]
            l=torch.nn.functional.interpolate(seen['lap'][0],e.shape[-2:],mode='bilinear',align_corners=False)
            self.assertGreater(float(t.abs().sum()),0)
            expected=torch.cat([e,l,r],1) if variant=='head_shift_only' else torch.cat([e,l+t,r-t],1)
            torch.testing.assert_close(seen['fusion'][0],expected,rtol=0,atol=0)
            torch.testing.assert_close(seen['arm'][0][1],m-t if variant=='no_head_shift' else m,rtol=0,atol=0)
            torch.testing.assert_close(seen['head'][0],l if variant=='no_head_shift' else l+t,rtol=0,atol=0)
            self.assertEqual(len(seen['seg']),3);self.assertEqual(len({x[0] for x in seen['seg']}),1)
            torch.testing.assert_close(out[0],torch.nn.functional.interpolate(seen['seg'][0][1],(200,200),mode='bilinear',align_corners=False),rtol=0,atol=0)
            for values in seen.values():values.clear()
            model.eval()
            with torch.no_grad():prediction=model(torch.randn(1,3,200,200))
            self.assertEqual(len(prediction),1);self.assertEqual(seen['head'],[])
            self.assertEqual(len(seen['t']),0 if variant=='head_shift_only' else 1)
            for h in handles:h.remove()

    def test_common_and_transport_initialization_all_seeds(self):
        for seed in (1337,2027,3407):
            result=audit(seed,['coupled_transport',*E1B_MATRIX]);self.assertEqual(result['status'],'PASS')
            for key in ('common_tensor_collection_sha256','transport_initial_tensor_sha256'):
                self.assertEqual(len({r[key] for r in result['checks'].values()}),1)

    def test_protocol_compatibility_and_mode_lock(self):
        original=base();self.assertEqual(protocol_errors(original),[])
        self.assertEqual(protocol_errors(dict(original,experiment_stage='E1-A',seed=2027,split_seed=1337)),[])
        for seed in (1337,2027,3407):
            for v in E1B_MATRIX:
                cfg=dict(original,experiment_stage='E1-B',matrix=E1B_MATRIX,variant=v,seed=seed,split_seed=1337)
                self.assertEqual(protocol_errors(cfg),[]);self.assertEqual(mode_errors(cfg,v),[])
                self.assertTrue(mode_errors(cfg,'coupled_transport'))
                self.assertTrue(protocol_errors(dict(cfg,split_seed=3407)))
        from ..scripts.server_preflight import preflight
        shared=Path('research/industrial_defect_english/configs/e1b_protocol_v1.json').resolve()
        cfg.update(shared_protocol_file=str(shared),shared_protocol_sha256='f'*64)
        with tempfile.TemporaryDirectory() as td,patch('research.industrial_defect_english.scripts.server_preflight.git_identity',return_value={'errors':[]}):
            path=Path(td)/'config.json';path.write_text(json.dumps(cfg))
            report=preflight('a'*40,path,False,cfg['variant'],'synthetic')
            self.assertTrue(any('shared protocol path/SHA mismatch' in x for x in report['blockers']))
            cfg['shared_protocol_sha256']=sha256(shared);path.write_text(json.dumps(cfg))
            report=preflight('a'*40,path,False,cfg['variant'],'synthetic')
            self.assertTrue(any('frozen asset SHAs' in x for x in report['blockers']))

    def test_nine_configs_preserve_assets_and_distinct_hashes(self):
        with tempfile.TemporaryDirectory(prefix='e1b paths ') as td:
            root=Path(td);common=base()
            for key,count in [('train_manifest',2903),('validation_manifest',726),('quarantine_manifest',1)]:
                p=root/key;p.write_text('{}\n'*count);common[key]=str(p);common[key+'_sha256']=sha256(p)
            for key,obj in [('condition_file',{}),('split_audit',{'status':'FROZEN_EXPLORATORY_D4_ONLY','seed':1337,'actual_train':2903,'actual_validation':726,'quarantined_train_count':1})]:
                p=root/key;p.write_text(json.dumps(obj));common[key]=str(p);common['condition_sha256' if key=='condition_file' else key+'_sha256']=sha256(p)
            refs=[];preserved={}
            for seed in (1337,2027,3407):
                d=root/str(seed);d.mkdir();refs.append(d);c=dict(common,seed=seed)
                if seed!=1337:c.update(experiment_stage='E1-A',split_seed=1337)
                p=d/'executed_config.json';p.write_text(json.dumps(c));preserved[p]=p.read_bytes()
                (d/'run.json').write_text(json.dumps({'seed':seed,'variant':'coupled_transport','status':'complete','result_status':'COMPLETE','completed_updates':10000,'planned_updates':10000,'source_commit_sha':E0_COMMIT if seed==1337 else E1A_COMMIT,'common_initial_tensor_sha256':'c'*64,'executed_config_sha256':sha256(p)}))
            hashes=prepare(refs,root/'new');self.assertEqual(len(hashes),9);self.assertEqual(len(set(hashes.values())),9)
            batches={}
            for name,h in hashes.items():
                p=root/'new'/name;self.assertEqual(sha256(p),h);c=json.loads(p.read_text())
                self.assertEqual(protocol_errors(c),[]);self.assertFalse(c['budget']['author_approved']);self.assertFalse(c['gpu_training_approved'])
                self.assertEqual(c['train_manifest_sha256'],common['train_manifest_sha256']);self.assertEqual(c['split_seed'],1337)
                batches.setdefault(c['seed'],set()).add(c['batch_id'])
            self.assertEqual(len({next(iter(x)) for x in batches.values()}),3)
            self.assertTrue(all(p.read_bytes()==b for p,b in preserved.items()))
            with self.assertRaises(FileExistsError):prepare(refs,root/'new')
            (refs[0]/'executed_config.json').write_text('{}')
            with self.assertRaises(ValueError):prepare(refs,root/'bad')

    def test_light_export_e1b_and_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root/'run')
            cfg={'optimizer_updates':10000,'experiment_stage':'E1-B'}
            for name in ('config_snapshot.json','executed_config.json'):(root/'run'/name).write_text(json.dumps(cfg))
            p=root/'run/run.json';m=json.loads(p.read_text());m.update(variant='head_shift_only',config=cfg,executed_config=cfg,config_sha256=sha256(root/'run/config_snapshot.json'),executed_config_sha256=sha256(root/'run/executed_config.json'));p.write_text(json.dumps(m))
            result=export(root/'run',root/'small');self.assertEqual(result['result_status'],'INCOMPLETE');self.assertEqual(result['experiment_stage'],'E1-B')
            self.assertTrue(verify_package(root/'small'));self.assertFalse(json.loads((root/'small/mechanism_response.json').read_text())['inference_transport_applicable'])
            with self.assertRaisesRegex(ValueError,'Incomplete endpoint'):read_package(root/'small')

    def test_same_seed_summary_and_wrong_reference_rejection(self):
        cfg=base();cfg.update({k:'b'*64 for k in ('train_manifest_sha256','validation_manifest_sha256','condition_sha256','split_audit_sha256','quarantine_manifest_sha256')})
        def make(seed,v,full=False):
            c=dict(cfg,seed=seed,split_seed=1337)
            if not full:c.update(experiment_stage='E1-B',variant=v,reference_coupled_transport={'source_commit_sha':E0_COMMIT if seed==1337 else E1A_COMMIT,'executed_config_sha256':'d'*64})
            m={'seed':seed,'variant':v,'source_commit_sha':(E0_COMMIT if seed==1337 else E1A_COMMIT) if full else 'a'*40,'common_initial_tensor_sha256':'c'*64,'transport_initial_tensor_sha256':'f'*64,'executed_config_sha256':'d'*64,'config_sha256':'e'*64}
            return {'meta':m,'config':c,'metrics':{'miou_including_background':.8 if full else .78,'foreground_miou':.7 if full else .68},'class_iou':{i:.8 if full else .78 for i in range(4)},'cases':{('synthetic',1):{'gt_area':'4','area_group':'small','complexity_group':'large','iou':'.8' if full else '.78'}},'errors':{},'conditions':{},'response':[],'mechanism':None,'runtime':{}}
        records={};refs=[];abls=[]
        for seed in (1337,2027,3407):
            name=f'full{seed}';refs.append(name);records[name]=make(seed,'coupled_transport',True)
            for v in E1B_MATRIX:name=f'{v}{seed}';abls.append(name);records[name]=make(seed,v)
        with patch('research.industrial_defect_english.scripts.summarize_e1b.read_package',side_effect=lambda p:records[p]):
            result=summarize(abls,refs,'a'*40)
            self.assertAlmostEqual(result['paired']['stage1_only']['miou_delta']['mean'],-.02)
            self.assertAlmostEqual(result['paired']['head_shift_only']['miou_delta']['std_sample'],0.)
            self.assertAlmostEqual(result['absolute']['coupled_transport']['class_iou'][1]['mean'],.8)
            self.assertAlmostEqual(result['paired']['stage1_only']['class_iou_delta'][1]['mean'],-.02)
            self.assertEqual(result['paired']['no_head_shift']['paired_strata']['class1_area_small'][0]['n'],1)
            with self.assertRaises(ValueError):summarize(abls[:-1],refs,'a'*40)
            records[abls[0]]['config']['reference_coupled_transport']['executed_config_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'paired reference'):summarize(abls,refs,'a'*40)

if __name__=='__main__':unittest.main()

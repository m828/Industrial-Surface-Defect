"""Synthetic CPU-only E1-C checks. No real data, training or GPU access."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from ..models import build_model
from ..models.transport import RelationTransport
from ..scripts.handoff_common import E1C_MATRIX,protocol_errors,mode_errors,sha256
from ..scripts.initialization_audit import audit
from ..scripts.prepare_e1b import E0_COMMIT,E1A_COMMIT
from ..scripts.prepare_e1c import prepare,check_reference_binding
from ..scripts.export_e0_artifacts import export,verify_package
from ..scripts.summarize_e1c import summarize,read_package

SEEDS=(1337,2027,3407)


def base():return json.loads(Path('research/industrial_defect_english/configs/e0_exploratory_d4_runtime_template_v1.json').read_text())


def capture(model,x,training):
    model.train(training);seen={'up':[],'lap':[],'transport':[],'fusion':[],'d':[],'arm':[],'head':[],'seg':[]}
    handles=[model.conv_up.register_forward_hook(lambda m,a,o:seen['up'].append(o)),
      model.laplacian.register_forward_hook(lambda m,a,o:seen['lap'].append(o)),
      model.transport.register_forward_hook(lambda m,a,o:seen['transport'].append((id(m),a,o))),
      model.edge_fusion.register_forward_pre_hook(lambda m,a:seen['fusion'].append(a[0])),
      model.edge_fusion.register_forward_hook(lambda m,a,o:seen['d'].append(o)),
      model.arm2.register_forward_pre_hook(lambda m,a:seen['arm'].append(a)),
      model.seg_head_detailloss.register_forward_pre_hook(lambda m,a:seen['head'].append(a[0])),
      model.seg_heads.register_forward_hook(lambda m,a,o:seen['seg'].append((id(m),o)))]
    torch.manual_seed(777)
    with torch.no_grad():seen['out']=model(x)
    for handle in handles:handle.remove()
    return seen


class E1CTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(8);x=torch.randn(2,3,200,200);cls.models={};cls.paths={};cls.initial={}
        for seed in SEEDS:cls.initial[seed]=audit(seed,['coupled_transport',*E1C_MATRIX])
        for variant in ('coupled_transport','recomputed_final_transport'):
            model=build_model(variant)
            with torch.no_grad():model.transport.predict[-1].bias.fill_(.2)
            cls.models[variant]=model
            cls.paths[variant]={True:capture(model,x,True),False:capture(model,x[:1],False)}

    def test_factory_builds_existing_mode(self):
        m=self.models['recomputed_final_transport']
        self.assertEqual(m.variant,'recomputed_final_transport')
        self.assertEqual(sum(isinstance(x,RelationTransport) for x in m.modules()),1)

    def test_first_stage_transport_and_fusion_identical(self):
        for training in (True,False):
            a=self.paths['coupled_transport'][training];b=self.paths['recomputed_final_transport'][training]
            for key in ('fusion','d'):
                torch.testing.assert_close(a[key][0],b[key][0],rtol=0,atol=0)
            for index in (0,1):
                torch.testing.assert_close(a['transport'][0][1][index],b['transport'][0][1][index],rtol=0,atol=0)
                torch.testing.assert_close(a['transport'][0][2][index],b['transport'][0][2][index],rtol=0,atol=0)
            e,r,_=b['up'];l=b['transport'][0][1][0];t=b['transport'][0][2][0]
            torch.testing.assert_close(b['fusion'][0],torch.cat([e,l+t,r-t],1),rtol=0,atol=0)

    def test_final_transport_recomputed_from_current_d_m(self):
        for training in (True,False):
            b=self.paths['recomputed_final_transport'][training];self.assertEqual(len(b['transport']),2)
            _,args,(tf,_)=b['transport'][1];m=b['up'][2]
            self.assertIs(args[0],b['d'][0]);self.assertIs(args[1],m)
            self.assertFalse(torch.equal(tf,b['transport'][0][2][0]))
            torch.testing.assert_close(b['arm'][0][1],m-tf,rtol=0,atol=0)
            a=self.paths['coupled_transport'][training];self.assertEqual(len(a['transport']),1)
            torch.testing.assert_close(a['arm'][0][1],a['up'][2]-a['transport'][0][2][0],rtol=0,atol=0)

    def test_predictor_parameters_shared_with_both_gradient_paths(self):
        for training in (True,False):
            calls=self.paths['recomputed_final_transport'][training]['transport']
            self.assertEqual({c[0] for c in calls},{id(self.models['recomputed_final_transport'].transport)})
        module=RelationTransport()
        with torch.no_grad():module.predict[-1].bias.fill_(.2)
        a=torch.randn(2,8,8,8,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
        t,_=module(a,b);tf,_=module(a+t,b-t)
        (t.square().mean()+tf.square().mean()).backward()
        self.assertTrue(torch.isfinite(module.predict[-1].bias.grad).all())
        self.assertGreater(float(module.predict[-1].bias.grad.abs().sum()),0)
        self.assertIsNotNone(a.grad);self.assertIsNotNone(b.grad)

    def test_auxiliary_detail_input_keeps_first_t(self):
        for variant in self.models:
            seen=self.paths[variant][True];l=seen['transport'][0][1][0];t=seen['transport'][0][2][0]
            torch.testing.assert_close(seen['head'][0],l+t,rtol=0,atol=0)
        torch.testing.assert_close(self.paths['coupled_transport'][True]['head'][0],self.paths['recomputed_final_transport'][True]['head'][0],rtol=0,atol=0)

    def test_training_eval_and_final_main_output(self):
        for variant in self.models:
            for training in (True,False):
                seen=self.paths[variant][training]
                self.assertEqual(len(seen['out']),4 if training else 1)
                self.assertEqual(len(seen['head']),1 if training else 0)
                self.assertEqual(len(seen['seg']),3 if training else 1)
                self.assertEqual(len({s[0] for s in seen['seg']}),1)
                torch.testing.assert_close(seen['out'][0],torch.nn.functional.interpolate(seen['seg'][0][1],(200,200),mode='bilinear',align_corners=False),rtol=0,atol=0)
                self.assertTrue(all(torch.isfinite(o).all() for o in seen['out']))

    def test_parameter_count_and_parameter_names_unchanged(self):
        a,b=self.models.values()
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        self.assertEqual({n:tuple(p.shape) for n,p in a.named_parameters()},{n:tuple(p.shape) for n,p in b.named_parameters()})
        self.assertEqual(sum(p.numel() for p in b.transport.parameters()),305)

    def references(self,root):
        common=base();assets={}
        for key,n in [('train_manifest',2903),('validation_manifest',726),('quarantine_manifest',1)]:
            p=root/key;p.write_text('{}\n'*n);common[key]=str(p);common[key+'_sha256']=sha256(p);assets[key+'_sha256']=sha256(p)
        for key,obj in [('condition_file',{}),('split_audit',{'status':'FROZEN_EXPLORATORY_D4_ONLY','seed':1337,'actual_train':2903,'actual_validation':726,'quarantined_train_count':1})]:
            p=root/key;p.write_text(json.dumps(obj));common[key]=str(p)
            h='condition_sha256' if key=='condition_file' else key+'_sha256';common[h]=sha256(p);assets[h]=sha256(p)
        refs=[]
        for seed in SEEDS:
            d=root/str(seed);d.mkdir();refs.append(d);cfg=dict(common,seed=seed)
            if seed!=1337:cfg.update(experiment_stage='E1-A',split_seed=1337)
            p=d/'executed_config.json';p.write_text(json.dumps(cfg))
            (d/'run.json').write_text(json.dumps({'seed':seed,'variant':'coupled_transport','status':'complete','result_status':'COMPLETE','completed_updates':10000,'planned_updates':10000,'source_commit_sha':E0_COMMIT if seed==1337 else E1A_COMMIT,'common_initial_tensor_sha256':self.initial[seed]['checks']['coupled_transport']['common_tensor_collection_sha256'],'executed_config_sha256':sha256(p),'config_sha256':'e'*64}))
        return refs,assets

    def test_three_config_generation_sha_and_original_files_preserved(self):
        with tempfile.TemporaryDirectory(prefix='e1c paths ') as td:
            root=Path(td);refs,assets=self.references(root)
            frozen={p:p.read_bytes() for d in refs for p in d.iterdir()}
            with patch('research.industrial_defect_english.scripts.prepare_e1c.FROZEN_ASSETS',assets):
                hashes=prepare(refs,root/'new');self.assertEqual(len(hashes),3);self.assertEqual(len(set(hashes.values())),3)
                for name,h in hashes.items():
                    p=root/'new'/name;c=json.loads(p.read_text());self.assertEqual(h,sha256(p));self.assertEqual(protocol_errors(c),[])
                    self.assertEqual(c['split_seed'],1337);self.assertEqual(c['matrix'],E1C_MATRIX)
                    self.assertFalse(c['gpu_training_approved']);self.assertFalse(c['budget']['author_approved'])
                    self.assertEqual(check_reference_binding(c)['seed'],c['seed'])
                self.assertTrue(all(p.read_bytes()==v for p,v in frozen.items()))
                with self.assertRaises(FileExistsError):prepare(refs,root/'new')
            # Production constants must reject these synthetic dataset identities.
            with self.assertRaisesRegex(ValueError,'frozen E0/E1-A identity'):prepare(refs,root/'must_not_exist')
            self.assertFalse((root/'must_not_exist').exists())

    def test_seed_and_initialization_binding(self):
        for seed,r in self.initial.items():
            self.assertEqual(r['status'],'PASS')
            for field in ('common_tensor_collection_sha256','transport_initial_tensor_sha256','total_parameters'):
                self.assertEqual(len({x[field] for x in r['checks'].values()}),1)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);refs,assets=self.references(root)
            with patch('research.industrial_defect_english.scripts.prepare_e1c.FROZEN_ASSETS',assets):
                prepare(refs,root/'new');p=root/'new/e1c_recomputed_final_transport_s1337.json';c=json.loads(p.read_text())
                c['seed']=2027
                with self.assertRaisesRegex(ValueError,'reference identity'):check_reference_binding(c)
                meta=refs[0]/'run.json';r=json.loads(meta.read_text());r['common_initial_tensor_sha256']='f'*64;meta.write_text(json.dumps(r))
                with self.assertRaisesRegex(ValueError,'common initialization'):prepare(refs,root/'bad')

    def test_e1c_preflight_gate_rejects_bad_sha_reference_and_cli_mode(self):
        from ..scripts.server_preflight import preflight
        shared=Path('research/industrial_defect_english/configs/e1c_protocol_v1.json').resolve()
        cfg=dict(base(),experiment_stage='E1-C',matrix=E1C_MATRIX,variant=E1C_MATRIX[0],seed=1337,split_seed=1337,shared_protocol_file=str(shared),shared_protocol_sha256='f'*64)
        with tempfile.TemporaryDirectory() as td,patch('research.industrial_defect_english.scripts.server_preflight.git_identity',return_value={'errors':[]}):
            p=Path(td)/'config.json';p.write_text(json.dumps(cfg))
            r=preflight('a'*40,p,False,cfg['variant'],'synthetic')
            self.assertEqual(r['status'],'BLOCKED');self.assertEqual(r['optimizer_steps'],0)
            self.assertTrue(any('shared protocol path/SHA' in x for x in r['blockers']))
            cfg['shared_protocol_sha256']=sha256(shared);cfg.update(json.loads(shared.read_text())['frozen_asset_sha256']);p.write_text(json.dumps(cfg))
            with patch('research.industrial_defect_english.scripts.prepare_e1c.check_reference_binding',side_effect=ValueError('synthetic wrong reference')):
                r=preflight('a'*40,p,False,cfg['variant'],'synthetic')
            self.assertTrue(any('wrong reference' in x for x in r['blockers']))
        self.assertTrue(mode_errors(cfg,'coupled_transport'))

    def synthetic_endpoint(self,root,seed,full=False):
        from .test_handoff import fixture
        fixture(root,'complete',10000);cfg=base();cfg.update(seed=seed,split_seed=1337)
        if not full:cfg.update(experiment_stage='E1-C',matrix=E1C_MATRIX,variant=E1C_MATRIX[0])
        elif seed!=1337:cfg['experiment_stage']='E1-A'
        assets={k:'b'*64 for k in ('train_manifest_sha256','validation_manifest_sha256','quarantine_manifest_sha256','split_audit_sha256','condition_sha256')}
        cfg.update(assets,expected_source_sha256='b'*64)
        cfg.update({k:None for k in ('train_manifest','validation_manifest','quarantine_manifest','condition_file','split_audit')})
        init=self.initial[seed]['checks']['coupled_transport']
        cfg.update(reference_common_initial_tensor_sha256=init['common_tensor_collection_sha256'],reference_transport_initial_tensor_sha256=init['transport_initial_tensor_sha256'])
        for name in ('config_snapshot.json','executed_config.json'):(root/name).write_text(json.dumps(cfg))
        source=(E0_COMMIT if seed==1337 else E1A_COMMIT) if full else 'a'*40
        meta=json.loads((root/'run.json').read_text());meta.update(source_commit_sha=source,seed=seed,variant='coupled_transport' if full else E1C_MATRIX[0],executed_config=cfg,config=cfg,config_sha256=sha256(root/'config_snapshot.json'),executed_config_sha256=sha256(root/'executed_config.json'),common_initial_tensor_sha256=init['common_tensor_collection_sha256'],transport_initial_tensor_sha256=init['transport_initial_tensor_sha256'],parameter_count=init['total_parameters'],elapsed_wall_seconds=1.,peak_cuda_bytes=0)
        (root/'endpoint.pt').write_text('SYNTHETIC FILE FOR HASH TEST ONLY, NOT A REAL CHECKPOINT')
        (root/'updates.jsonl').write_text(''.join(json.dumps({'completed_updates':i})+'\n' for i in range(1,10001)))
        cases=[{'sample_id':f'synthetic{i:04d}','class_id':c,'gt_area':6 if c<3 else 7,'area_group':'small','complexity_group':'large','iou':.8 if full else .78} for i in range(726) for c in (1,2,3)]
        response=[{'beta_mean':.1,'transport_rms':.2,'rho_difference':.3}]*(1 if full else 2)
        v={'confusion':[[726*x for x in row] for row in [[10,1,0,0],[1,5,0,0],[0,0,6,0],[0,0,0,7]]], 'iou':[.8,.7,1.,1.], 'miou_including_background':.8 if full else .78,'foreground_miou':.7 if full else .68,'cases':cases,'transport_response':response*726}
        (root/'metrics.jsonl').write_text(json.dumps({'optimizer_updates':10000,'validation':v,'training_transport_response':response})+'\n')
        for name,key in [('endpoint.pt','checkpoint_sha256'),('updates.jsonl','updates_sha256'),('metrics.jsonl','metrics_sha256')]:meta[key]=sha256(root/name)
        (root/'run.json').write_text(json.dumps(meta));pf=root/'preflight.json';pf.write_text(json.dumps({'status':'GO_READY','config_sha256':meta['config_sha256'],'git':{'source_commit_sha':source}}))
        return cfg,assets,pf

    def test_light_exports_and_paired_summary_including_incomplete_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);refs=[];fresh=[];metadata={}
            for seed in SEEDS:
                for full in (True,False):
                    path=root/f'{seed}_{full}';cfg,assets,pf=self.synthetic_endpoint(path,seed,full)
                    if not full:
                        cfg['reference_coupled_transport']=metadata[seed]
                        for name in ('config_snapshot.json','executed_config.json'):(path/name).write_text(json.dumps(cfg))
                        m=json.loads((path/'run.json').read_text());m.update(config=cfg,executed_config=cfg,config_sha256=sha256(path/'config_snapshot.json'),executed_config_sha256=sha256(path/'executed_config.json'));(path/'run.json').write_text(json.dumps(m))
                        report=json.loads(pf.read_text());report['config_sha256']=m['config_sha256'];pf.write_text(json.dumps(report))
                    else:
                        m=json.loads((path/'run.json').read_text());metadata[seed]={'source_commit_sha':m['source_commit_sha'],'executed_config_sha256':m['executed_config_sha256']}
                    package=root/(path.name+'_light');result=export(path,package,pf);self.assertTrue(result['fixed_endpoint_comparable']);self.assertTrue(verify_package(package))
                    (refs if full else fresh).append(package)
            with patch('research.industrial_defect_english.scripts.summarize_e1c.FROZEN_ASSETS',assets):
                result=summarize(fresh,refs,'a'*40)
                self.assertAlmostEqual(result['paired_aggregate']['miou_delta']['mean'],.02)
                self.assertEqual([r['seed'] for r in result['paired']],list(SEEDS))
                self.assertEqual(len(result['per_run']),6);self.assertTrue(result['paired_strata'])
                self.assertEqual(result['per_run'][1]['response']['final']['n_calls'],726)
                with self.assertRaises(ValueError):summarize(fresh[:-1],refs,'a'*40)
            # Re-export a deliberately stopped fixture; it must not become a fixed endpoint.
            path=root/'1337_False';m=json.loads((path/'run.json').read_text());m.update(status='stopped',completed_updates=9999);(path/'run.json').write_text(json.dumps(m))
            incomplete=root/'incomplete';r=export(path,incomplete,path/'preflight.json');self.assertFalse(r['fixed_endpoint_comparable'])
            with self.assertRaisesRegex(ValueError,'Incomplete endpoint'):read_package(incomplete)

    def test_e0_e1a_e1b_compatibility_and_e1c_matrix_isolation(self):
        from ..scripts.handoff_common import MATRIX,E1B_MATRIX
        self.assertEqual(MATRIX,['formal_a2ms','coupled_transport','capacity_control'])
        self.assertEqual(E1B_MATRIX,['stage1_only','head_shift_only','no_head_shift'])
        cfg=base();self.assertEqual(protocol_errors(cfg),[])
        self.assertEqual(protocol_errors(dict(cfg,experiment_stage='E1-A',seed=2027,split_seed=1337)),[])
        for v in E1B_MATRIX:
            self.assertEqual(protocol_errors(dict(cfg,experiment_stage='E1-B',matrix=E1B_MATRIX,variant=v,split_seed=1337)),[])
        for seed in SEEDS:
            c=dict(cfg,experiment_stage='E1-C',matrix=E1C_MATRIX,variant=E1C_MATRIX[0],seed=seed,split_seed=1337)
            self.assertEqual(protocol_errors(c),[]);self.assertEqual(mode_errors(c,E1C_MATRIX[0]),[])
            self.assertTrue(protocol_errors(dict(c,matrix=MATRIX)));self.assertTrue(protocol_errors(dict(c,split_seed=2027)))

if __name__=='__main__':unittest.main()

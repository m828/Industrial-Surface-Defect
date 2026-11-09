"""SYNTHETIC fixtures only: patched counts/hashes are never production inputs."""
import contextlib,copy,hashlib,json,os,random,subprocess,tempfile,textwrap,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,TensorDataset
from ..scripts import ksdd2_contract as contract
from ..scripts.ksdd2_task import Task,require_execution
from ..scripts.ksdd2_support import BINARY_LOSS_ID,MATRIX,initialization_audit
from ..scripts.handoff_common import sha256,source_snapshot_sha256
from ..scripts.training_loop import execute_updates
from ..scripts.prepare_ksdd2 import prepare
from ..scripts.export_e0_artifacts import export,verify_package
from ..scripts.summarize_ksdd2 import summarize


class TinyNet(torch.nn.Module):
    """Loop/optimizer fixture; real four-model forward checks remain in test_ksdd2."""
    def __init__(self,classes=2):
        super().__init__();self.seg=torch.nn.Conv2d(3,classes,1);self.detail=torch.nn.Conv2d(3,1,1)
    def forward(self,x):
        logits=self.seg(x)
        return [logits,logits,logits,self.detail(x)] if self.training else [logits]


class Oracle(torch.nn.Module):
    """Known synthetic RGB signal for coordinate tests, not a measured model."""
    def forward(self,x):return [torch.stack((.5-x[:,0],x[:,0]-.5),1)]


def fixture(root):
    data=root/'synthetic data';frozen=root/'synthetic freeze';runs=root/'synthetic runs'
    data.mkdir();frozen.mkdir();runs.mkdir();rows={'train':[],'validation':[]}
    shapes=[(632,230),(668,240),(672,256),(641,237),(633,231),(643,233)]
    for i,(h,w) in enumerate(shapes):
        mask=np.zeros((h,w),np.uint8)
        if i in (0,4):mask[0,0]=255;mask[-1,-1]=255
        elif i==1:mask[40:90,40:90]=255
        image=np.zeros((h,w,3),np.uint8);image[:,:,0]=mask
        ip=data/f'image{i}.png';mp=data/f'mask{i}.png';Image.fromarray(image).save(ip);Image.fromarray(mask).save(mp)
        split='train' if i<4 else 'validation'
        rows[split].append(dict(sample_id=f'SYNTHETIC_{i}',image_relative=ip.name,mask_relative=mp.name,
          image_sha256=sha256(ip),mask_sha256=sha256(mp),native_hw=[h,w],gt_positive=bool(mask.any()),original_split='train',group_id=f'source{i}'))
    objects={'preprocessing':{'input_hw':[672,256],'num_classes':2,'mask_values':[0,255]},
      'area':{'p33':contract.P33,'source_positive_n':2},
      'split_report':{'freeze_id':contract.FREEZE_ID,'train_n':4,'validation_n':2,'train_positive_n':2,'validation_positive_n':1},
      'fingerprint':{'dataset_id':'KSDD2','scope':'SYNTHETIC FIXTURE, NOT REAL DATA'}}
    for key,split in [('train_manifest','train'),('validation_manifest','validation')]:
        (frozen/contract.ASSETS[key]).write_text(''.join(json.dumps(r)+'\n' for r in rows[split]))
    for key,value in objects.items():(frozen/contract.ASSETS[key]).write_text(json.dumps(value))
    (frozen/'SHA256SUMS.txt').write_text(''.join(sha256(frozen/name)+'  '+name+'\n' for key,name in contract.ASSETS.items() if key!='checksum_list'))
    cfg=json.loads(Path('research/industrial_defect_english/configs/ksdd2_runtime_template_v1.json').read_text())
    cfg.update(data_root=str(data),runs_root=str(runs),batch_id='SYNTHETIC_KSDD2_s1337',
      assets={k:str(frozen/name) for k,name in contract.ASSETS.items()},expected_source_sha256=source_snapshot_sha256(),
      schema=json.loads(Path('research/industrial_defect_english/configs/ksdd2_schema_map_example.json').read_text()))
    cfg['asset_sha256']={k:sha256(p) for k,p in cfg['assets'].items()}
    cfg['budget'].update(max_wall_seconds_per_run=300,max_batch_gpu_hours=1,max_run_bytes=50_000_000,min_free_bytes=1)
    return cfg,rows,objects


@contextlib.contextmanager
def synthetic_contract(cfg,objects):
    hashes={k:cfg['asset_sha256'][k] for k in contract.FROZEN_HASHES}
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(contract,'FROZEN_HASHES',hashes))
        stack.enter_context(patch.object(contract,'COUNTS',{'train':(4,2),'validation':(2,1)}))
        stack.enter_context(patch.object(contract,'TRAIN_AREA_COUNTS',(1,1,2)))
        stack.enter_context(patch.object(contract,'JSON_SEMANTICS',objects))
        stack.enter_context(patch('research.industrial_defect_english.scripts.prepare_ksdd2.FROZEN_HASHES',hashes))
        stack.enter_context(patch('research.industrial_defect_english.scripts.summarize_ksdd2.COUNTS',{'train':(4,2),'validation':(2,1)}))
        yield


def approved(cfg):
    c=copy.deepcopy(cfg);c.update(scientific_protocol_approved=True,gpu_training_approved=True)
    c['budget']['author_approved']=True;c['stopping']['author_approved']=True;return c


class KSDD2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.initial=initialization_audit(1337)

    def test_01_manifest_identity_counts_and_raw_shape(self):
        with tempfile.TemporaryDirectory() as td:
            c,_,o=fixture(Path(td))
            with synthetic_contract(c,o):
                rows,report=contract.data_contract(c)
                self.assertEqual(len(rows['train']),4);self.assertEqual(report['training_area_groups'],{'small':1,'other':1,'normal':2})
                self.assertTrue(report['raw_files_checked'])
            sums=Path(c['assets']['checksum_list'])
            sums.write_text(''.join(c['asset_sha256'][k]+'  '+str(Path(c['assets'][k]).resolve())+'\n' for k in contract.ASSETS if k!='checksum_list'))
            c['asset_sha256']['checksum_list']=sha256(sums)
            with synthetic_contract(c,o):self.assertEqual(contract.data_contract(c)[1]['cohorts']['train']['n'],4)
            with self.assertRaisesRegex(ValueError,'manifest identity'):contract.data_contract(c)

    def test_02_nonfixed_native_sizes_boundary_and_empty_mask(self):
        with tempfile.TemporaryDirectory() as td:
            c,_,o=fixture(Path(td))
            with synthetic_contract(c,o):rows,_=contract.data_contract(c)
            ds=contract.KSDD2Data(rows['train']);a=ds[0];edge=ds[2]
            self.assertEqual(tuple(a['image'].shape),(3,672,256));self.assertEqual(a['native_hw'].tolist(),[632,230])
            self.assertEqual(int(a['labels'].sum()),2);self.assertEqual(int(a['valid'].sum()),632*230)
            self.assertEqual(int(a['labels'][0,0]),1);self.assertEqual(int(a['labels'][631,229]),1)
            self.assertFalse(a['labels'][~a['valid']].any());self.assertEqual(int(edge['valid'].sum()),672*256)
            self.assertFalse(edge['labels'].any());self.assertEqual(float(a['image'].max()),1.)

    def test_03_flip_before_padding_uses_same_geometry_and_rng(self):
        with tempfile.TemporaryDirectory() as td:
            c,_,o=fixture(Path(td))
            with synthetic_contract(c,o):rows,_=contract.data_contract(c)
            a=contract.KSDD2Data(rows['train'],True,SimpleNamespace(random=lambda:0.))[0]
            self.assertEqual(int(a['labels'][0,229]),1);self.assertEqual(float(a['image'][0,0,229]),1.)
            self.assertFalse(a['valid'][:,230:].any())
            with self.assertRaises(ValueError):contract.KSDD2Data(rows['train'],True)

    def test_04_valid_mask_loss_and_padding_gradients(self):
        with tempfile.TemporaryDirectory() as td:
            c,_,o=fixture(Path(td))
            with synthetic_contract(c,o):rows,_=contract.data_contract(c)
            batch=next(iter(DataLoader(contract.KSDD2Data(rows['train']),batch_size=2)))
            model=TinyNet();loss,parts,info=Task(c).training_loss(model,batch,'cpu');loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertEqual(len(parts),4)
            self.assertEqual(info['valid_pixels'],632*230+668*240)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_05_empty_gt_fallback_logged(self):
        from ..scripts.ksdd2_support import binary_joint_loss
        y=torch.zeros(2,32,48,dtype=torch.long);v=torch.ones_like(y,dtype=torch.bool)
        logits=torch.stack((torch.full_like(y,1000.,dtype=torch.float),torch.full_like(y,-1000.,dtype=torch.float)),1).requires_grad_()
        loss,_,d=binary_joint_loss([logits]*3+[torch.zeros(2,1,32,48,requires_grad=True)],y,v,[10,1,3,3],BINARY_LOSS_ID)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(d['empty_hard_pixel_ce_fallback_routes'],[0,2])

    def test_06_native_validation_and_empty_gt_errors_exclude_padding(self):
        with tempfile.TemporaryDirectory() as td:
            c,_,o=fixture(Path(td))
            with synthetic_contract(c,o):rows,_=contract.data_contract(c)
            loader=DataLoader(contract.KSDD2Data(rows['validation']),batch_size=1)
            result=Task(c).validate(Oracle(),loader,device='cpu')
            self.assertEqual(result['foreground_iou'],1.);self.assertEqual(result['tp'],2)
            self.assertEqual(sum(sum(r) for r in result['confusion_matrix']),633*231+643*233)
            self.assertEqual(result['empty_gt_n'],1);self.assertEqual(result['both_empty_n'],1)
            self.assertIsNone(result['cases'][1]['iou']);self.assertEqual(result['cases'][0]['native_h'],633)
            class AllForeground(torch.nn.Module):
                def forward(self,x):return [torch.stack((torch.zeros_like(x[:,0]),torch.ones_like(x[:,0])),1)]
            bad=Task(c).validate(AllForeground(),loader,device='cpu')
            self.assertEqual(bad['empty_gt_image_fpr'],1.);self.assertEqual(bad['empty_gt_false_positive_pixels'],643*233)

    def test_07_shared_loop_one_cpu_update_and_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,_,o=fixture(root)
            with synthetic_contract(c,o):rows,_=contract.data_contract(c)
            c.update(optimizer_updates=1,validation_every=1);c['budget']['max_run_bytes']=1_000_000_000
            out=root/'SYNTHETIC_UPDATE';out.mkdir()
            sample=torch.Generator().manual_seed(1337);val_rng=torch.Generator().manual_seed(1338);aug=random.Random(1337)
            train=DataLoader(contract.KSDD2Data(rows['train'],True,aug),batch_size=2,shuffle=True,generator=sample)
            val=DataLoader(contract.KSDD2Data(rows['validation']),batch_size=1,generator=val_rng)
            from ..models import build_model
            model=build_model('coupled_transport',2,1337)
            before=list(model.seg_heads.parameters())[-1].detach().clone()
            opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=2e-6)
            meta={'completed_updates':0}
            execute_updates(model,opt,train,val,c,out,meta,time.perf_counter(),c['budget'],sample,val_rng,aug,iter(train),task=Task(c),device='cpu')
            self.assertEqual(meta['completed_updates'],1);self.assertFalse(torch.equal(before,list(model.seg_heads.parameters())[-1]))
            self.assertTrue((out/'recovery.pt').is_file());self.assertTrue((out/'endpoint.pt').is_file())
            row=json.loads((out/'metrics.jsonl').read_text());self.assertIn('loss_diagnostics',row)
            self.assertEqual(row['validation']['coordinate_system'],'native_no_padding')

    def task_preflight(self,c,path):
        from ..scripts.server_preflight import preflight
        path.write_text(json.dumps(c));run_id=f"{c['batch_id']}:formal_a2ms:s{c['seed']}"
        with patch('research.industrial_defect_english.scripts.ksdd2_task.git_identity',return_value={'errors':[],'repo_root':str(Path.cwd())}), \
             patch('research.industrial_defect_english.scripts.ksdd2_task.verify_loaded_sources',return_value={'synthetic_fixture':True}), \
             patch('research.industrial_defect_english.scripts.ksdd2_task.initialization_audit',return_value=self.initial), \
             patch('research.industrial_defect_english.scripts.server_preflight.environment',return_value={'cuda_available':True,'gpu_usage_query':{'returncode':0}}), \
             patch.object(torch.optim.Adam,'step',side_effect=AssertionError('PREFLIGHT MUST NOT STEP')):
            return preflight('a'*40,path,True,'formal_a2ms',run_id)

    def test_08_preflight_unauthorized_gpu_and_budget_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,_,o=fixture(root)
            with synthetic_contract(c,o):r=self.task_preflight(c,root/'config.json')
            self.assertEqual(r['status'],'BLOCKED');self.assertIn('approval_missing:gpu_training_approved',r['blockers'])
            self.assertIn('approval_missing:budget',r['blockers']);self.assertEqual(r['optimizer_steps'],0)
            with self.assertRaises(ValueError):require_execution(c,SimpleNamespace(authorize_gpu=True,source_commit='a'*40,run_id='x',variant='formal_a2ms'))

    def test_09_preflight_rejects_each_asset_sha_and_wrong_source(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,_,o=fixture(root);c=approved(c)
            with synthetic_contract(c,o):
                for key in contract.ASSETS:
                    bad=copy.deepcopy(c);bad['asset_sha256'][key]='f'*64
                    r=self.task_preflight(bad,root/'config.json');self.assertEqual(r['status'],'BLOCKED')
                    self.assertTrue(any('SHA' in x for x in r['blockers']))
                bad=copy.deepcopy(c);bad['expected_source_sha256']='f'*64
                self.assertIn('source_snapshot_mismatch',self.task_preflight(bad,root/'config.json')['blockers'])

    def test_10_synthetic_ready_gate_and_output_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,_,o=fixture(root);c=approved(c)
            with synthetic_contract(c,o):
                r=self.task_preflight(c,root/'config.json');self.assertEqual(r['status'],'GO_READY');self.assertFalse(r['training_authorized'])
                with patch('research.industrial_defect_english.scripts.ksdd2_task.os.access',return_value=False):
                    blocked=self.task_preflight(c,root/'config.json')
                    self.assertTrue(any('not writable' in x for x in blocked['blockers']))
                p=Path(c['runs_root'])/c['batch_id']/'formal_a2ms_s1337';p.mkdir(parents=True)
                self.assertIn('run_output_already_exists',self.task_preflight(c,root/'config.json')['blockers'])

    def test_11_data_pairing_path_escape_shape_and_threshold_guards(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,_,o=fixture(root)
            with synthetic_contract(c,o):
                bad=copy.deepcopy(c);bad['schema']['area']['p33']='/source_positive_n'
                with self.assertRaisesRegex(ValueError,'Frozen JSON'):contract.data_contract(bad)
                path=Path(c['data_root'])/'mask0.png';Image.fromarray(np.zeros((2,2),np.uint8)).save(path)
                with self.assertRaisesRegex(ValueError,'Raw file SHA'):contract.data_contract(c)
            with self.assertRaises(ValueError):contract.field({},None)

    def test_12_prepare_three_pending_configs_without_refreeze(self):
        with tempfile.TemporaryDirectory(prefix='ksdd2 paths ') as td:
            root=Path(td);c,_,o=fixture(root);schema=root/'schema.json';schema.write_text(json.dumps(c['schema']))
            original={p:p.read_bytes() for p in Path(c['assets']['checksum_list']).parent.iterdir()}
            with synthetic_contract(c,o):
                hashes=prepare(Path(c['assets']['checksum_list']).parent,c['data_root'],c['runs_root'],schema,root/'configs')
                self.assertEqual(len(hashes),3);self.assertEqual(len(set(hashes.values())),3)
                for name,h in hashes.items():
                    p=root/'configs'/name;cfg=json.loads(p.read_text());self.assertEqual(sha256(p),h)
                    self.assertFalse(cfg['gpu_training_approved']);self.assertFalse(cfg['scientific_protocol_approved'])
                    self.assertEqual(cfg['input_hw'],[672,256]);self.assertEqual(cfg['num_classes'],2)
                with self.assertRaises(FileExistsError):prepare(Path(c['assets']['checksum_list']).parent,c['data_root'],c['runs_root'],schema,root/'configs')
            self.assertTrue(all(p.read_bytes()==v for p,v in original.items()))

    def test_13_dataset_dispatch_rejects_unknown_and_neu_masquerade(self):
        import sys
        from ..scripts.train_e0_ready import main
        from ..scripts.server_preflight import preflight
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'config.json';p.write_text(json.dumps({'dataset_id':'ksdd2'}))
            args=['runner','--config',str(p),'--variant','formal_a2ms','--preflight-only']
            with patch.object(sys,'argv',args),patch('research.industrial_defect_english.scripts.ksdd2_task.run',return_value='task-selected') as run:
                self.assertEqual(main(),'task-selected');run.assert_called_once()
            with patch.dict(sys.modules,{'research.industrial_defect_english.scripts.ksdd2_task':None}):
                self.assertEqual(preflight('a'*40,p,False)['status'],'BLOCKED')
            p.write_text(json.dumps({'dataset_id':'unknown'}));self.assertEqual(preflight('a'*40,p,False)['status'],'BLOCKED')
            p.write_text('[]');self.assertEqual(preflight('a'*40,p,False)['status'],'BLOCKED')
            self.assertTrue(contract.protocol_errors({'dataset_id':'neu','num_classes':4}))

    def synthetic_run(self,root,c,seed,variant):
        root.mkdir();c=copy.deepcopy(c);c.update(seed=seed,batch_id=f'SYNTHETIC_s{seed}')
        for name in ('executed_config.json','config_snapshot.json'):(root/name).write_text(json.dumps(c))
        cohorts,_=contract.data_contract(c);loader=DataLoader(contract.KSDD2Data(cohorts['validation']),batch_size=1)
        v=Task(c).validate(Oracle(),loader,device='cpu')
        calls=0 if variant=='formal_a2ms' else (2 if variant=='recomputed_final_transport' else 1)
        v['transport_response']=[{'beta_mean':.01,'transport_rms':.001,'rho_difference':.02}]*len(cohorts['validation'])*calls
        (root/'metrics.jsonl').write_text(json.dumps({'optimizer_updates':10000,'validation':v,'training_transport_response':[]})+'\n')
        (root/'updates.jsonl').write_text(''.join(json.dumps({'completed_updates':i})+'\n' for i in range(1,10001)))
        (root/'endpoint.pt').write_text('SYNTHETIC HASH FIXTURE, NOT A REAL CHECKPOINT')
        pf=root/'synthetic_preflight.json';pf.write_text(json.dumps({'dataset_id':'ksdd2','status':'GO_READY','source_commit_sha':'a'*40,'config_sha256':sha256(root/'config_snapshot.json'),'asset_sha256':c['asset_sha256']}))
        m={'dataset_id':'ksdd2','variant':variant,'seed':seed,'executed_config':c,'status':'complete','completed_updates':10000,'planned_updates':10000,
           'source_commit_sha':'a'*40,'config_sha256':sha256(root/'config_snapshot.json'),'executed_config_sha256':sha256(root/'executed_config.json'),
           'preflight_sha256':sha256(pf),'common_initial_tensor_sha256':hashlib.sha256(str(seed).encode()).hexdigest(),
           'transport_initial_tensor_sha256':None if variant=='formal_a2ms' else 'b'*64,'parameter_count':10,'elapsed_wall_seconds':1.,'peak_cuda_bytes':0}
        for name,key in [('endpoint.pt','checkpoint_sha256'),('metrics.jsonl','metrics_sha256'),('updates.jsonl','updates_sha256')]:m[key]=sha256(root/name)
        (root/'run.json').write_text(json.dumps(m));return pf

    def test_14_binary_export_schema_and_safety_whitelist(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,_,o=fixture(root)
            with synthetic_contract(c,o):
                pf=self.synthetic_run(root/'run',c,1337,'recomputed_final_transport');m=export(root/'run',root/'light',pf)
                self.assertEqual(m['schema'],'ksdd2_light_v1');self.assertTrue(m['fixed_endpoint_comparable'])
                self.assertTrue(verify_package(root/'light'));self.assertFalse((root/'light/endpoint.pt').exists())
                with (root/'light/confusion_matrix.csv').open() as f:self.assertEqual(len(f.read().splitlines()),5)
                with self.assertRaises(FileExistsError):export(root/'run',root/'light',pf)
                (root/'light/weights.pt').write_text('SYNTHETIC FORBIDDEN')
                with self.assertRaises(ValueError):verify_package(root/'light')

    def test_15_twelve_endpoint_summary_pairs_seeds_and_keeps_na(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,_,o=fixture(root);packages=[]
            with synthetic_contract(c,o):
                for seed in (1337,2027,3407):
                    for model in MATRIX:
                        run=root/f'{model}_{seed}';pf=self.synthetic_run(run,c,seed,model)
                        package=root/(run.name+'_light');export(run,package,pf);packages.append(package)
                result=summarize(packages,'a'*40)
                self.assertEqual(len(result['per_run']),12)
                self.assertEqual(result['paired']['formal_a2ms']['foreground_iou_direction']['tied'],3)
                self.assertEqual([r['seed'] for r in result['paired']['capacity_control']['per_seed']],[1337,2027,3407])
                self.assertEqual(result['aggregate']['coupled_transport']['other_positive_macro_iou']['missing_n'],3)
                with self.assertRaises(ValueError):summarize(packages[:-1],'a'*40)
                with self.assertRaises(ValueError):summarize(packages,'f'*40)

    def test_16_incomplete_export_and_summary_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,_,o=fixture(root)
            with synthetic_contract(c,o):
                run=root/'run';pf=self.synthetic_run(run,c,1337,'coupled_transport');p=run/'run.json';m=json.loads(p.read_text());m.update(status='stopped',completed_updates=9999);p.write_text(json.dumps(m))
                light=root/'light';meta=export(run,light,pf);self.assertFalse(meta['fixed_endpoint_comparable'])
                with self.assertRaisesRegex(ValueError,'INCOMPLETE'):summarize([light],'a'*40)

    def test_17_neu_shared_loop_one_step_matches_published_source(self):
        from ..models.losses import joint_loss_with_protocol,loss_component_names
        from ..scripts.e0_contract import check_budget,require_finite,require_finite_loss,response_stats,summarize_validation
        source=subprocess.check_output(['git','show','8abd026bb6b0fce051196379604f9c0ccabd6aac:research/industrial_defect_english/scripts/train_e0_ready.py'],text=True)
        a=source.index('        training_response=[];');b=source.index("        metadata.update(status='complete'",a)
        old=textwrap.dedent(source[a:b]);namespace=dict(torch=torch,json=json,os=os,random=random,time=time,np=np,
          joint_loss_with_protocol=joint_loss_with_protocol,loss_component_names=loss_component_names,check_budget=check_budget,
          require_finite=require_finite,require_finite_loss=require_finite_loss,response_stats=response_stats,summarize_validation=summarize_validation)
        signature='model,optimizer,train_loader,val_loader,cfg,run,metadata,started,runtime_budget,sample_rng,val_rng,augmentation_rng,it,validate,conditions'
        exec(compile('def original('+signature+'):\n'+textwrap.indent(old,'    '),'published_loop_fixture','exec'),namespace)
        torch.manual_seed(3);template=TinyNet(4);x=torch.randn(2,3,32,32);y=torch.randint(0,4,(2,32,32))
        cfg={'optimizer_updates':1,'validation_every':1,'experiment_stage':'E0','loss_weights':[10,1,3,3],
             'loss_protocol_id':'neu_a2ms_mask_aware_candidate_v1','num_classes':4}
        v={'confusion':[[2 if i==j else 1 for j in range(4)] for i in range(4)],'cases':[],'transport_response':[]}
        with tempfile.TemporaryDirectory() as td,patch.object(torch.Tensor,'cuda',lambda self:self),patch('torch.cuda.synchronize'),patch('torch.cuda.get_rng_state_all',return_value=[]):
            states=[];losses=[]
            for legacy in (True,False):
                model=copy.deepcopy(template);opt=torch.optim.Adam(model.parameters(),lr=1e-4);sample=torch.Generator().manual_seed(1337);vr=torch.Generator().manual_seed(1338);aug=random.Random(1337)
                train=DataLoader(TensorDataset(x,y),batch_size=2,shuffle=True,generator=sample);out=Path(td)/str(legacy);out.mkdir()
                args=(model,opt,train,None,cfg,out,{'completed_updates':0},time.perf_counter(),{'max_wall_seconds_per_run':60,'max_run_bytes':10000000},sample,vr,aug,iter(train))
                validate=lambda *args:copy.deepcopy(v)
                if legacy:namespace['original'](*args,validate,{})
                else:execute_updates(*args,legacy_validate=validate,conditions={})
                states.append(copy.deepcopy(model.state_dict()));losses.append(json.loads((out/'metrics.jsonl').read_text())['loss'])
            self.assertEqual(losses[0],losses[1])
            for key in states[0]:torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)

    def test_18_unknown_task_protocol_and_fixed_data_settings_rejected(self):
        c=json.loads(Path('research/industrial_defect_english/configs/ksdd2_runtime_template_v1.json').read_text())
        for key,value in [('num_classes',4),('input_hw',[640,256]),('loss_protocol_id','neu_a2ms_mask_aware_candidate_v1'),('mask_encoding','zero_one'),('evaluation_split','test'),('optimizer_updates',9999)]:
            bad=copy.deepcopy(c);bad[key]=value;self.assertIn('ksdd2_protocol:'+key,contract.protocol_errors(bad))
        with self.assertRaises(ValueError):require_execution(c,SimpleNamespace(authorize_gpu=False,source_commit='a'*40,run_id='x',variant='coupled_transport'))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'config.json';p.write_text(json.dumps(c))
            import sys
            result=subprocess.run([sys.executable,'-m','research.industrial_defect_english.scripts.train_e0_ready','--config',str(p),'--variant','formal_a2ms','--preflight-only'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('BLOCKED',result.stdout)

    def test_19_four_real_models_at_frozen_input_size(self):
        from ..models import build_model
        for variant in MATRIX:
            model=build_model(variant,2,1337).eval()
            with torch.no_grad():out=model(torch.zeros(1,3,672,256))
            self.assertEqual(tuple(out[0].shape),(1,2,672,256));self.assertTrue(torch.isfinite(out[0]).all())

    def test_20_startup_allocation_failure_keeps_zero_update_record(self):
        from ..scripts.ksdd2_task import run,binding
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,_,o=fixture(root);c=approved(c);path=root/'config.json';path.write_text(json.dumps(c))
            rid=f"{c['batch_id']}:formal_a2ms:s1337"
            report=dict(binding('a'*40,path,c,'formal_a2ms',rid),status='GO_READY',blockers=[],optimizer_steps=0,training_authorized=False,
              checks={'initialization_fairness':self.initial},source_snapshot_sha256=c['expected_source_sha256'],source_integrity={'synthetic':True},environment={})
            pf=root/'synthetic_preflight.json';pf.write_text(json.dumps(report))
            args=SimpleNamespace(config=str(path),variant='formal_a2ms',authorize_gpu=True,source_commit='a'*40,run_id=rid,preflight_only=False,preflight_report=str(pf),preflight_sha256=sha256(pf))
            with synthetic_contract(c,o),patch('research.industrial_defect_english.scripts.ksdd2_task.preflight',return_value=report), \
                 patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.manual_seed_all'), \
                 patch('torch.cuda.get_device_name',return_value='SYNTHETIC'),patch('torch.cuda.max_memory_allocated',return_value=0), \
                 patch('research.industrial_defect_english.models.build_model',side_effect=RuntimeError('SYNTHETIC allocation failure')):
                with self.assertRaisesRegex(RuntimeError,'SYNTHETIC allocation'):run(args)
            record=json.loads((Path(c['runs_root'])/c['batch_id']/'formal_a2ms_s1337/run.json').read_text())
            self.assertEqual(record['result_status'],'INCOMPLETE');self.assertEqual(record['completed_updates'],0)
            self.assertIsNone(record['common_initial_tensor_sha256']);self.assertIn('allocation',record['reason'])

if __name__=='__main__':unittest.main()

"""KSDD2 frozen-data identity and explicit task contract; no training side effects."""
import copy,json,re
from pathlib import Path,PurePosixPath
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .handoff_common import sha256
from .ksdd2_support import MATRIX,SEEDS,BINARY_LOSS_ID,decode_mask,pad_native

FREEZE_ID='KSDD2_DATA_FROZEN_V1'
PROTOCOL_ID='ksdd2_binary_fixed_endpoint_candidate_v1'
P33=0.009815324081003446
FROZEN_HASHES={'train_manifest':'dd7d50550c23887da15234f425378a0ce4e07974d0ebe533a13254db457c29c3',
 'validation_manifest':'482a8c16e4c096f8a3a3035783fb519c2570555b768dfb04da537105c9a58ee6'}
COUNTS={'train':(1865,197),'validation':(466,49)}
TRAIN_AREA_COUNTS=(66,131,1668)
ASSETS={'train_manifest':'dev_train.jsonl','validation_manifest':'dev_validation.jsonl',
 'preprocessing':'preprocessing_protocol.json','area':'area_threshold_p33.json',
 'fingerprint':'dataset_fingerprint.json','split_report':'split_freeze_report.json','checksum_list':'SHA256SUMS.txt'}
JSON_SEMANTICS={'preprocessing':{'input_hw':[672,256],'num_classes':2,'mask_values':[0,255]},
 'area':{'p33':P33,'source_positive_n':197},
 'split_report':{'freeze_id':FREEZE_ID,'train_n':1865,'validation_n':466,'train_positive_n':197,'validation_positive_n':49},
 'fingerprint':{'dataset_id':'KSDD2'}}


def field(obj,spec):
    """Explicit JSON pointers; a pointer list combines e.g. separate H and W fields."""
    if isinstance(spec,list):return [field(obj,p) for p in spec]
    if spec=='':return obj  # RFC6901: explicitly map a scalar JSON document root.
    if not isinstance(spec,str) or not spec.startswith('/'):raise ValueError('Explicit JSON-pointer field mapping required')
    for token in spec[1:].split('/'):
        token=token.replace('~1','/').replace('~0','~')
        obj=obj[int(token)] if isinstance(obj,list) else obj[token]
    return obj


def config(path):
    p=Path(path).resolve();c=copy.deepcopy(json.loads(p.read_text()));base=p.parent
    for key in ('data_root','runs_root'):
        if c.get(key):c[key]=str((base/Path(c[key])).resolve())
    for key,value in c.get('assets',{}).items():c['assets'][key]=str((base/Path(value)).resolve())
    return c


def protocol_errors(c):
    want={'dataset_id':'ksdd2','task_protocol_id':PROTOCOL_ID,'data_freeze_id':FREEZE_ID,'num_classes':2,
      'matrix':list(MATRIX),'input_hw':[672,256],'mask_encoding':'zero_255','rng_policy':'isolated_v1',
      'initialization':'from_scratch_common_seed','optimizer_updates':10000,'batch_size':16,
      'optimizer':{'name':'Adam','lr':1e-4,'weight_decay':2e-6},'scheduler':'constant','validation_every':1000,
      'loss_protocol_id':BINARY_LOSS_ID,'loss_weights':[10,1,3,3],'class_weights':None,
      'ohem':{'threshold':.7,'min_kept':100000,'empty_policy':'mean_ce_on_valid'},'detail_valid_radius':4,
      'augmentation':'paired_horizontal_flip_before_padding_p05_v1','checkpoint_selection':'fixed final optimizer update 10000',
      'evaluation_split':'development_validation','image_normalization':'RGB/255','resize':False,'crop':False,
      'padding':'right_bottom_replicate','prediction':'two_logits_argmax_no_postprocess'}
    errors=['ksdd2_protocol:'+k for k,v in want.items() if c.get(k)!=v]
    if type(c.get('seed')) is not int or c['seed'] not in SEEDS:errors.append('ksdd2_protocol:seed')
    if c.get('performance_early_stop') is not False or c.get('response_early_stop') is not False:errors.append('ksdd2_protocol:early_stop')
    if c.get('stopping',{}).get('policy')!='safety_and_resource_only':errors.append('ksdd2_protocol:stopping')
    if not re.fullmatch('[A-Za-z0-9_.-]+',c.get('batch_id') or '') or c.get('batch_id') in ('.','..'):errors.append('ksdd2_protocol:batch_id')
    for key,value in FROZEN_HASHES.items():
        if c.get('asset_sha256',{}).get(key)!=value:errors.append('ksdd2_frozen_SHA:'+key)
    return errors


def approvals(c):
    errors=[]
    for key in ('scientific_protocol_approved','gpu_training_approved'):
        if c.get(key) is not True:errors.append('approval_missing:'+key)
    if c.get('budget',{}).get('author_approved') is not True:errors.append('approval_missing:budget')
    if c.get('stopping',{}).get('author_approved') is not True:errors.append('approval_missing:safety_stopping')
    return errors


def read_checksums(path):
    result={}
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        h,name=line.split(maxsplit=1);name=name.strip().removeprefix('*').removeprefix('./')
        p=PurePosixPath(name)
        # This is a read-only digest index, not an extraction/path traversal API.
        # Absolute names are accepted only by exact matching to configured assets below.
        if not re.fullmatch('[0-9a-f]{64}',h) or '..' in p.parts or name in result:raise ValueError('Unsafe checksum list')
        result[name]=h
    return result


def data_contract(c,check_raw=True):
    assets=c.get('assets',{});hashes=c.get('asset_sha256',{});maps=c.get('schema',{})
    if set(assets)!=set(ASSETS) or set(hashes)!=set(ASSETS):raise ValueError('All seven frozen asset paths/SHAs are required')
    for key,p in assets.items():
        if not re.fullmatch('[0-9a-f]{64}',hashes[key] or '') or sha256(p)!=hashes[key]:raise ValueError('Frozen asset SHA mismatch: '+key)
    for key,h in FROZEN_HASHES.items():
        if hashes[key]!=h:raise ValueError('Author-frozen manifest identity mismatch: '+key)
    sums=read_checksums(assets['checksum_list'])
    for key,name in ASSETS.items():
        if key!='checksum_list':
            listed=[sums[k] for k in (name,str(Path(assets[key]).resolve())) if k in sums]
            if not listed or any(h!=hashes[key] for h in listed):raise ValueError('SHA256SUMS/source mismatch: '+name)
    verified_semantics={}
    for asset,expected in JSON_SEMANTICS.items():
        obj=json.loads(Path(assets[asset]).read_text())
        if obj is None or obj=={} or obj==[]:raise ValueError('Empty frozen metadata asset: '+asset)
        if asset=='area' and maps.get(asset,{}).get('p33') is None:raise ValueError('Explicit P33 JSON pointer required')
        for key,value in expected.items():
            pointer=maps.get(asset,{}).get(key)
            if pointer is None:continue  # No guessed schema: bytes remain locked and author-reviewed in config.
            if field(obj,pointer)!=value:raise ValueError('Frozen JSON identity mismatch: '+asset+':'+key)
            verified_semantics[asset+':'+key]=value
    root=Path(c.get('data_root') or '').resolve()
    if not c.get('data_root') or not root.is_dir():raise ValueError('Explicit readable dataset root required')
    normalized={};report={'data_freeze_id':FREEZE_ID,'asset_sha256':hashes,'raw_files_checked':check_raw,'cohorts':{},
      'mapped_json_semantics':verified_semantics,'metadata_scope':'all seven asset SHAs pinned; only explicitly mapped JSON fields semantically decoded; runtime preprocessing fixed by author contract'}
    m=maps.get('manifest',{})
    required=('sample_id','image_relative','mask_relative','image_sha256','mask_sha256','native_hw','gt_positive','original_split')
    for split,key in (('train','train_manifest'),('validation','validation_manifest')):
        raw=[json.loads(x) for x in Path(assets[key]).read_text().splitlines() if x.strip()];rows=[]
        for item in raw:
            row={k:field(item,m.get(k)) for k in required};row['sample_id']=str(row['sample_id'])
            row['group_id']=field(item,m['group_id']) if m.get('group_id') else None
            if row['original_split']!='train' or type(row['gt_positive']) is not bool:raise ValueError('Official test entry or invalid GT presence in development manifest')
            for kind in ('image','mask'):
                relative=PurePosixPath(row[kind+'_relative'])
                path=(root/relative).resolve()
                if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):raise ValueError('Raw asset path escapes dataset root')
                row[kind]=str(path)
                if not re.fullmatch('[0-9a-f]{64}',row[kind+'_sha256'] or ''):raise ValueError('Missing per-file identity')
                if check_raw and sha256(path)!=row[kind+'_sha256']:raise ValueError('Raw file SHA mismatch: '+row['sample_id'])
            if check_raw:
                with Image.open(row['image']) as im:hw=[im.height,im.width]
                with Image.open(row['mask']) as im:y=decode_mask(np.array(im),'zero_255')
                if hw!=row['native_hw'] or tuple(y.shape)!=tuple(hw) or hw[0]>672 or hw[1]>256:raise ValueError('Native shape/frozen pad contract mismatch')
                if bool(y.any())!=row['gt_positive']:raise ValueError('GT positivity differs from frozen record')
                row['gt_fraction']=float(y.sum())/y.size
            rows.append(row)
        n,p=COUNTS[split]
        if len(rows)!=n or sum(r['gt_positive'] for r in rows)!=p:raise ValueError('Frozen cohort count mismatch: '+split)
        for identity in ('sample_id','image','mask'):
            if len({r[identity] for r in rows})!=len(rows):raise ValueError('Duplicate pairing identity: '+identity)
        normalized[split]=rows;report['cohorts'][split]={'n':n,'positive':p,'normal':n-p}
    for key in ('sample_id','image','mask','image_sha256'):
        if {r[key] for r in normalized['train']}&{r[key] for r in normalized['validation']}:raise ValueError('Development overlap: '+key)
    groups=[{r['group_id'] for r in normalized[s] if r['group_id'] is not None} for s in ('train','validation')]
    if groups[0]&groups[1]:raise ValueError('Known source group overlap')
    report['parent_scope']='FROZEN_GROUP_METADATA_NOT_COLLECTION_INDEPENDENCE_PROOF' if any(groups) else 'UNKNOWN_NOT_PROVIDED'
    if check_raw:
        small=sum(r['gt_positive'] and r['gt_fraction']<=P33 for r in normalized['train'])
        actual=(small,COUNTS['train'][1]-small,COUNTS['train'][0]-COUNTS['train'][1])
        if actual!=TRAIN_AREA_COUNTS:raise ValueError('Frozen training small/other/normal group mismatch')
        report['training_area_groups']=dict(zip(('small','other','normal'),actual))
    return normalized,report


class KSDD2Data(Dataset):
    def __init__(self,rows,augment=False,rng=None):
        if augment and rng is None:raise ValueError('Isolated augmentation RNG required')
        self.rows=rows;self.augment=augment;self.rng=rng
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]
        with Image.open(r['image']) as im:x=np.array(im.convert('RGB'),copy=True)
        with Image.open(r['mask']) as im:y=decode_mask(np.array(im),'zero_255')
        if list(y.shape)!=r['native_hw'] or tuple(x.shape[:2])!=tuple(y.shape):raise ValueError('Image/GT shape changed since preflight')
        if self.augment and self.rng.random()<.5:x=x[:,::-1].copy();y=y[:,::-1].copy()
        image,label,valid=pad_native(x,y,[672,256])
        return {'image':image,'labels':label,'valid':valid,'native_hw':torch.tensor(y.shape),
                'sample_id':r['sample_id'],'gt_positive':bool(y.any())}

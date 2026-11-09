"""Offline binary-task primitives. No training, dataset scan or implicit defaults.

These helpers do not relax the NEU runner/preflight gates. KSDD2 task dispatch
uses an independent frozen-data and candidate-training contract.
"""
import copy,hashlib,math,random,re
import numpy as np
import torch
from torch.nn import functional as F
from ..models.losses import boundary_target
from ..models.metrics import case_metrics

MATRIX=('formal_a2ms','capacity_control','coupled_transport','recomputed_final_transport')
SEEDS=(1337,2027,3407)
BINARY_LOSS_ID='ksdd2_binary_masked_ohem_detail_candidate_v1'


def initialization_audit(seed):
    from .initialization_audit import audit
    result=audit(seed,MATRIX,num_classes=2)
    return dict(result,num_classes=2,dataset='KSDD2')


def decode_mask(mask,encoding):
    a=np.asarray(mask)
    if a.ndim!=2 or not np.issubdtype(a.dtype,np.integer):raise ValueError('Expected a 2D integer binary mask')
    allowed={'zero_one':{0,1},'zero_255':{0,255}}
    if encoding not in allowed or not set(np.unique(a))<=allowed[encoding]:raise ValueError('Explicit mask encoding mismatch')
    return (a!=0).astype(np.int64)


def pad_native(rgb,labels,input_hw):
    """No resize/crop. Return RGB/255, indexed GT and independent valid mask."""
    a=np.asarray(rgb);y=np.asarray(labels)
    if a.ndim!=3 or a.shape[2]!=3 or a.dtype!=np.uint8 or y.shape!=a.shape[:2]:raise ValueError('RGB/mask shape or type mismatch')
    if not set(np.unique(y))<={0,1}:raise ValueError('Labels must already be decoded to 0/1')
    h,w=y.shape;hh,ww=input_hw
    if hh%32 or ww%32 or min(hh,ww)<224 or h>hh or w>ww:raise ValueError('Pad target must fit native image, be divisible by 32 and at least 224 per axis for SqueezeBodyEdge')
    image=torch.from_numpy(a.copy()).permute(2,0,1).float()/255
    image=F.pad(image,(0,ww-w,0,hh-h),mode='replicate')
    label=F.pad(torch.from_numpy(y.copy()).long(),(0,ww-w,0,hh-h),value=0)
    valid=torch.zeros((hh,ww),dtype=torch.bool);valid[:h,:w]=True
    return image,label,valid


def masked_ohem(logits,labels,valid,threshold=.7,min_kept=100000):
    """Candidate: mine valid pixels only; explicit finite CE fallback on empty hard set."""
    losses=F.cross_entropy(logits,labels,reduction='none')[valid]
    if not losses.numel():raise ValueError('No valid segmentation pixels')
    probabilities=logits.softmax(1).gather(1,labels[:,None]).squeeze(1)[valid]
    sorted_p,order=probabilities.detach().sort()
    cutoff=max(threshold,float(sorted_p[min(min_kept,sorted_p.numel()-1)]))
    keep=sorted_p<cutoff
    empty=not bool(keep.any())
    return (losses.mean() if empty else losses[order][keep].mean()),empty


def binary_joint_loss(outputs,labels,valid,weights,loss_protocol_id):
    """Task-adaptation candidate, NOT the historical NEU mask-aware protocol."""
    if loss_protocol_id!=BINARY_LOSS_ID:raise ValueError('Explicit KSDD2 candidate loss ID required')
    if len(outputs)!=4 or len(weights)!=4:raise ValueError('Three segmentation outputs and one detail output required')
    if labels.dtype!=torch.long or valid.dtype!=torch.bool or labels.shape!=valid.shape:raise ValueError('Invalid GT/validity tensors')
    if labels.min()<0 or labels.max()>1:raise ValueError('Binary labels only; padding is expressed by valid, not label255')
    expected=(labels.shape[0],2,*labels.shape[-2:])
    if any(tuple(o.shape)!=expected for o in outputs[:3]) or tuple(outputs[3].shape)!=(labels.shape[0],1,*labels.shape[-2:]):
        raise ValueError('Binary head shapes do not match GT')
    if not valid.flatten(1).any(1).all():raise ValueError('A sample has no valid pixels')
    target=F.one_hot(labels,2).permute(0,3,1,2).float()
    a,empty_a=masked_ohem(outputs[0],labels,valid)
    b=F.binary_cross_entropy_with_logits(outputs[1],target,reduction='none').mean(1)[valid].mean()
    c,empty_c=masked_ohem(outputs[2],labels,valid)
    # full/x2/x4 target depends on neighbors up to 4px from the output pixel.
    # Exclude uncertain outer/padding bands in the detail route, retain all
    # original pixels for segmentation. Never supervise padded pixels as negatives.
    v=F.pad(valid[:,None].float(),(4,4,4,4),value=0)
    detail_valid=(1-F.max_pool2d(1-v,9,1))[:,0].bool()
    counts=detail_valid.flatten(1).sum(1)
    if (counts==0).any():raise ValueError('Empty valid detail interior; choose a suitable input policy')
    edge=boundary_target(labels)[:,0];logits=outputs[3][:,0]
    bce=F.binary_cross_entropy_with_logits(logits,edge,reduction='none')[detail_valid].mean()
    p=logits.sigmoid()*detail_valid;t=edge*detail_valid
    dice=(1-(2*(p*t).flatten(1).sum(1)+1)/(p.flatten(1).sum(1)+t.flatten(1).sum(1)+1)).mean()
    d=bce+dice;parts=[a,b,c,d]
    return sum(w*x for w,x in zip(weights,parts)),parts,{'empty_hard_pixel_ce_fallback_routes':[i for i,e in ((0,empty_a),(2,empty_c)) if e],
      'valid_pixels':int(valid.sum()),'detail_valid_pixels':int(detail_valid.sum()),'loss_protocol_id':loss_protocol_id}


def safe_ratio(a,b):return a/b if b else None


def binary_metrics(cases,small_area_threshold=None):
    """Native-coordinate GT/pred masks; aggregate counts and explicit empty policy."""
    matrix=np.zeros((2,2),dtype=np.int64);rows=[];ids=set()
    for sid,gt,pred in cases:
        if sid in ids:raise ValueError('Duplicate case ID')
        ids.add(sid);g=np.asarray(gt);p=np.asarray(pred)
        if g.ndim!=2 or g.shape!=p.shape or not set(np.unique(g))<={0,1} or not set(np.unique(p))<={0,1}:
            raise ValueError('Expected paired native 0/1 masks')
        g=g.astype(np.int64);p=p.astype(np.int64)
        cm=np.bincount((2*g+p).ravel(),minlength=4).reshape(2,2);matrix+=cm
        row=case_metrics(torch.from_numpy(g),torch.from_numpy(p),2,sid)[0]
        row.update(gt_positive=bool(g.any()),pred_positive=bool(p.any()),pred_area=int(p.sum()),n_pixels=g.size)
        row['area_group']=('small' if row['gt_fraction']<=small_area_threshold else 'other') if row['gt_positive'] and small_area_threshold is not None else None
        rows.append(row)
    if not rows:raise ValueError('No evaluation cases')
    tn,fp,fn,tp=(int(v) for v in matrix.ravel());foreground_iou=safe_ratio(tp,tp+fp+fn)
    bg=safe_ratio(tn,tn+fp+fn);dice=safe_ratio(2*tp,2*tp+fp+fn)
    positive=[r for r in rows if r['gt_positive']];negative=[r for r in rows if not r['gt_positive']]
    valid=[r for r in rows if r['iou'] is not None]
    mean=lambda rr,key:sum(r[key] for r in rr)/len(rr) if rr else None
    normal_pixels=sum(r['n_pixels'] for r in negative);normal_fp=sum(r['pred_area'] for r in negative)
    small=[r for r in positive if r['area_group']=='small']
    other=[r for r in positive if r['area_group']=='other']
    return {'confusion_matrix':matrix.tolist(),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
      'foreground_iou':foreground_iou,'foreground_dice':dice,'background_iou':bg,
      'miou_including_background':(foreground_iou+bg)/2 if foreground_iou is not None and bg is not None else None,
      'precision':safe_ratio(tp,tp+fp),'recall':safe_ratio(tp,tp+fn),
      'case_macro_foreground_iou':mean(valid,'iou'),'case_macro_iou_n':len(valid),
      'gt_positive_macro_iou':mean(positive,'iou'),'gt_positive_macro_dice':mean(positive,'dice'),
      'gt_positive_n':len(positive),'empty_gt_n':len(negative),'both_empty_n':sum(not r['pred_positive'] for r in negative),
      'empty_gt_false_positive_images':sum(r['pred_positive'] for r in negative),
      'empty_gt_image_fpr':safe_ratio(sum(r['pred_positive'] for r in negative),len(negative)),
      'empty_gt_false_positive_pixels':normal_fp,'empty_gt_pixel_fpr':safe_ratio(normal_fp,normal_pixels),
      'small_positive_n':len(small),'small_positive_macro_iou':mean(small,'iou'),'small_positive_macro_dice':mean(small,'dice'),
      'other_positive_n':len(other),'other_positive_macro_iou':mean(other,'iou'),'other_positive_macro_dice':mean(other,'dice'),
      'cases':rows,'empty_policy':'both empty foreground IoU/Dice NA; one empty=0; all cases retained in pixel counts'}


def development_split(rows,official_train_ids,official_test_ids,validation_fraction=.2,seed=1337):
    """Pure metadata planning: group by supplied parent/pixel/D4 evidence, never files."""
    if not 0<validation_fraction<1:raise ValueError('Invalid validation fraction')
    if type(seed) is not int:raise ValueError('Explicit integer split seed required')
    records=sorted(copy.deepcopy(rows),key=lambda r:r['sample_id']);ids={r['sample_id'] for r in records}
    if len(ids)!=len(records) or ids!=set(official_train_ids) or ids & set(official_test_ids):raise ValueError('Official cohort identity mismatch')
    if any(r.get('original_split')!='train' or type(r.get('gt_positive')) is not bool for r in records):raise ValueError('Train-only labels and explicit GT presence required')
    if any(not r.get('image_pixel_sha256') for r in records):raise ValueError('Decoded-image identity required; filenames are not lineage evidence')
    if any(not re.fullmatch('[0-9a-f]{64}',r['image_pixel_sha256']) for r in records):raise ValueError('Invalid decoded-image SHA256')
    if any(r.get('parent_id') in ('UNKNOWN','unknown','UNRESOLVED') for r in records):raise ValueError('Unknown parent must be null, not a shared placeholder ID')
    parents=list(range(len(records)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    seen={}
    for i,row in enumerate(records):
        for field in ('parent_id','image_pixel_sha256','image_d4_canonical_sha256'):
            key=(field,row.get(field))
            if key[1] is None:continue
            if not key[1]:raise ValueError('Empty supplied grouping identity')
            if key in seen:parents[find(i)]=find(seen[key])
            else:seen[key]=i
    groups={}
    for i,row in enumerate(records):groups.setdefault(find(i),[]).append(row)
    items=list(groups.values());random.Random(seed).shuffle(items)
    totals=[sum(r['gt_positive'] for r in records),sum(not r['gt_positive'] for r in records)]
    if min(totals)==0:raise ValueError('Official training must contain positive and normal images')
    target=[n*validation_fraction for n in totals];counts=[0,0];selected=[]
    score=lambda c:sum(((c[i]-target[i])/max(target[i],1))**2 for i in range(2))
    while items:
        def objective(g):return score([counts[0]+sum(r['gt_positive'] for r in g),counts[1]+sum(not r['gt_positive'] for r in g)])
        i=min(range(len(items)),key=lambda j:objective(items[j]));g=items[i]
        if objective(g)>=score(counts):break
        selected.extend(items.pop(i));counts=[sum(r['gt_positive'] for r in selected),sum(not r['gt_positive'] for r in selected)]
    training=[r for g in items for r in g]
    if any(not any(r['gt_positive']==positive for r in part) for part in (training,selected) for positive in (True,False)):
        raise ValueError('Grouping prevents positive/normal coverage; do not retry seeds silently')
    for group in groups.values():
        group_id=min(r['sample_id'] for r in group)
        for r in group:r['group_id']=group_id
    fingerprint=lambda rr:hashlib.sha256(('\n'.join(sorted(r['sample_id'] for r in rr))+'\n').encode()).hexdigest()
    info={'status':'METADATA_PLAN_ONLY_NOT_REAL_DATA_FREEZE','seed':seed,'original_train_n':len(records),
      'development_train_n':len(training),'development_validation_n':len(selected),'group_n':len(groups),
      'train_positive_n':sum(r['gt_positive'] for r in training),'validation_positive_n':counts[0],
      'train_normal_n':sum(not r['gt_positive'] for r in training),'validation_normal_n':counts[1],
      'train_ids_sha256':fingerprint(training),'validation_ids_sha256':fingerprint(selected),
      'parent_metadata_status':'SUPPLIED_NOT_INDEPENDENTLY_VERIFIED' if all(r.get('parent_id') for r in records) else 'UNKNOWN_OR_PARTIAL',
      'scope':'known parent/exact content links only; no collection-source independence proof'}
    return sorted(training,key=lambda r:r['sample_id']),sorted(selected,key=lambda r:r['sample_id']),info


def area_threshold(training_rows):
    values=[float(r['gt_fraction']) for r in training_rows if r['gt_positive']]
    if not values or any(not math.isfinite(v) or not 0<v<=1 for v in values):raise ValueError('Training-positive GT fractions required')
    return {'small_area_fraction_p33':float(np.quantile(values,1/3,method='linear')),
            'source':'development training positives at native resolution only','n':len(values),'quantile_method':'numpy linear, q=1/3'}

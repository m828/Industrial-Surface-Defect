"""One-time dev split from readable original train pairs; no inference or training.

No fuzzy duplicate/source investigation. Groups are connected by exact decoded
D4 image equivalence and supplied parent IDs. Unknown lineage is never certified.
"""
import argparse,hashlib,json,csv,random
from pathlib import Path
import numpy as np
from PIL import Image

QUARANTINE_POLICY = 'neu_exact_d4_quarantine_candidate_v1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def d4_transforms(array):
    """Eight lossless orthogonal transforms; rotations are counterclockwise."""
    for k in range(4):
        yield ('identity' if k == 0 else f'r{90*k}'), np.rot90(array, k)
    flipped = np.flip(array, axis=1)
    for k in range(4):
        yield ('flip_h' if k == 0 else f'flip_h_r{90*k}'), np.rot90(flipped, k)


def pixel_hash(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def d4_match(a,b):
    """Verify pixels after candidate hashing and compare aligned GT separately."""
    with Image.open(a['image']) as im: first=np.array(im.convert('RGB'),copy=True)
    with Image.open(b['image']) as im: second=np.array(im.convert('RGB'),copy=True)
    with Image.open(a['mask']) as im: first_gt=np.array(im,copy=True)
    with Image.open(b['mask']) as im: second_gt=np.array(im,copy=True)
    image_transforms=dict(d4_transforms(first));gt_transforms=dict(d4_transforms(first_gt))
    matches=[]
    for name, transformed in image_transforms.items():
        if transformed.shape==second.shape and np.array_equal(transformed,second):
            gt=gt_transforms[name]
            unequal=int(np.count_nonzero(gt!=second_gt))
            matches.append({'transform_first_to_second':name,'gt_equal':unequal==0,
                            'gt_mismatch_pixels':unequal,'gt_total_pixels':int(gt.size),
                            'gt_mismatch_fraction':unequal/gt.size})
    return matches


def d4_evidence(train,test):
    """Compare candidate groups by canonical hash, then check decoded pixels."""
    by_hash={}
    for row in train+test:by_hash.setdefault(row['image_d4_canonical_sha256'],[]).append(row)
    within=[];cross=[]
    for rows in by_hash.values():
        for i,first in enumerate(rows):
            for second in rows[i+1:]:
                if first['original_split']==second['original_split']=='test':continue
                matches=d4_match(first,second)
                if not matches:continue
                rec={'first_split':first['original_split'],'first_sample_id':first['sample_id'],
                     'first_image_relative':first['original_image_relative'],
                     'first_image_sha256':first['image_sha256'],'first_mask_sha256':first['mask_sha256'],
                     'second_split':second['original_split'],'second_sample_id':second['sample_id'],
                     'second_image_relative':second['original_image_relative'],
                     'second_image_sha256':second['image_sha256'],'second_mask_sha256':second['mask_sha256'],
                     'matches':matches}
                if first['original_split']!=second['original_split']:
                    if first['sample_id']=='004097' and second['sample_id']=='000717':
                        rec['prior_record']='Previously reported train:004097/test:000717; this entry is based on the current decoded images.'
                    cross.append(rec)
                else:within.append(rec)
    return {'schema':'neu_d4_evidence_v1',
            'operations':'identity, counterclockwise 90/180/270 degrees, horizontal reflection and its 90/180/270 rotations',
            'verification':'Exact decoded RGB pixel equality; GT equality checked separately after the same transformation',
            'train_train_pairs':within,'train_test_pairs':cross,
            'limitations':'Exact D4 only; crop, resize, color changes and unknown original parentage are not excluded.'}


def scan(split,root,kind):
    rows=[]
    for line in Path(split).read_text().splitlines():
        if not line.strip():continue
        fields=line.split()
        if len(fields)!=2:raise ValueError('Expected mask path then image path per original list line')
        data_root=Path(root).resolve()
        mp,ip=[(data_root/f).resolve() for f in fields]
        if not all(p.is_relative_to(data_root) for p in (mp,ip)):raise ValueError('Split path escapes data root')
        if ip.stem!=mp.stem:raise ValueError(f'Image/mask ID mismatch: {ip} / {mp}')
        with Image.open(ip) as im: image=np.array(im.convert('RGB'),copy=True)
        with Image.open(mp) as im: mask=np.array(im,copy=True)
        if mask.ndim!=2 or image.shape[:2]!=mask.shape:raise ValueError(f'Image/mask shape mismatch: {ip}')
        if not np.issubdtype(mask.dtype,np.integer):raise ValueError(f'Non-integer mask: {mp}')
        values=set(np.unique(mask).tolist())
        if not values<={0,1,2,3}:raise ValueError(f'Invalid mask values: {mp}: {values}')
        pixel_sha=pixel_hash(image)
        canonical=min(pixel_hash(x) for _,x in d4_transforms(image))
        rows.append({'sample_id':ip.stem,'original_split':kind,'image':str(ip),'mask':str(mp),
                     'image_sha256':sha(ip),'mask_sha256':sha(mp),'image_pixel_sha256':pixel_sha,'image_d4_canonical_sha256':canonical,
                     'original_image_relative':fields[1],'original_mask_relative':fields[0],'shape':list(mask.shape),'present_classes':sorted(values),
                     'class_pixels':{str(c):int((mask==c).sum()) for c in range(4)}})
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate sample IDs within split')
    return rows


def choose_groups(groups,target,seed):
    # Deterministic, group-level greedy iterative stratification; no result-driven retries.
    rng=random.Random(seed);items=list(groups.values());rng.shuffle(items)
    total=sum(len(g) for g in items)
    all_counts=np.array([sum(c in r['present_classes'] for g in items for r in g) for c in (1,2,3)])
    wanted=all_counts*target/total; picked=[];n=0;counts=np.zeros(3)
    while n<target:
        candidates=[(i,g) for i,g in enumerate(items) if g and n+len(g)<=target]
        if not candidates:break
        def score(pair):
            _,g=pair;count=np.array([sum(c in r['present_classes'] for r in g) for c in (1,2,3)])
            return float(np.square((counts+count-wanted)/np.maximum(wanted,1)).sum())
        i,g=min(candidates,key=score);picked.extend(g);n+=len(g)
        counts+=np.array([sum(c in r['present_classes'] for r in g) for c in (1,2,3)]);items[i]=[]
    return [r for g in items for r in g],picked


def load_families(family_map,train):
    """Read supplied lineage; completeness does not establish its truth."""
    families={}
    if not family_map:return families
    with open(family_map) as f:
        for row in csv.DictReader(f):
            key=row['image_relative'];value=row['parent_id'].strip()
            if not value or key in families:raise ValueError('Empty parent ID or duplicate parent-map row')
            families[key]=value
    if set(families)!={r['original_image_relative'] for r in train}:
        raise ValueError('Parent-ID map must cover all training images')
    return families


def partition_training(train,evidence,families=None,quarantine_policy_id='disabled',evidence_sha256=None):
    """Partition full official training IDs; quarantine complete connected groups.

    Callers must derive evidence from the original images, not an unverified JSON.
    Parent information remains a supplied claim requiring separate acceptance.
    """
    if quarantine_policy_id not in ('disabled',QUARANTINE_POLICY):raise ValueError('Unknown quarantine policy ID')
    if evidence.get('schema')!='neu_d4_evidence_v1':raise ValueError('Unsupported D4 evidence schema')
    if quarantine_policy_id=='disabled' and evidence['train_test_pairs']:
        raise ValueError('Cross-original-split D4 image equivalence; quarantine is disabled')
    rows=[dict(r) for r in train];families=families or {}
    if any(r['original_split']!='train' for r in rows):raise ValueError('Only original training rows may be partitioned')
    indices={r['sample_id']:i for i,r in enumerate(rows)}
    if len(indices)!=len(rows):raise ValueError('Duplicate training IDs')
    if families and set(families)!={r['original_image_relative'] for r in rows}:
        raise ValueError('Parent-ID map must cover all training images')
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(i)]=find(j)
    seen={}
    if families:
        for i,row in enumerate(rows):
            name=families[row['original_image_relative']]
            if name in seen:union(i,seen[name])
            else:seen[name]=i
    for pair in evidence['train_train_pairs']:
        if pair['first_split']!='train' or pair['second_split']!='train':raise ValueError('Invalid within-train evidence split')
        i,j=indices[pair['first_sample_id']],indices[pair['second_sample_id']]
        if families and families[rows[i]['original_image_relative']]!=families[rows[j]['original_image_relative']]:
            raise ValueError('Supplied parent IDs conflict with confirmed D4 identity')
        union(i,j)
    groups={}
    for i,row in enumerate(rows):
        row['group_id']=str(find(i));groups.setdefault(str(find(i)),[]).append(row)
    direct={}
    for number,pair in enumerate(evidence['train_test_pairs']):
        if {pair['first_split'],pair['second_split']}!={'train','test'}:raise ValueError('Invalid cross-split evidence')
        key='first_sample_id' if pair['first_split']=='train' else 'second_sample_id'
        sid=pair[key]
        if sid not in indices:raise ValueError('Cross-split evidence references unknown train ID')
        direct.setdefault(sid,[]).append(number)
    affected={rows[indices[sid]]['group_id'] for sid in direct}
    eligible=[];quarantined=[]
    for group,members in groups.items():
        if group not in affected:eligible.extend(members);continue
        seed_ids=sorted(r['sample_id'] for r in members if r['sample_id'] in direct)
        match_indices=sorted({n for sid in seed_ids for n in direct[sid]})
        for row in members:
            row.update(quarantine_policy_id=quarantine_policy_id,
                       quarantine_reason='EXACT_D4_OFFICIAL_TEST_MATCH' if row['sample_id'] in direct else 'TRAIN_GROUP_CONNECTED_TO_EXACT_D4_TEST_MATCH',
                       quarantine_seed_train_ids=seed_ids,quarantine_match_indices=match_indices,
                       d4_evidence_sha256=evidence_sha256,
                       quarantine_matches=[evidence['train_test_pairs'][n] for n in match_indices])
            quarantined.append(row)
    original_ids=set(indices);eligible_ids={r['sample_id'] for r in eligible};quarantined_ids={r['sample_id'] for r in quarantined}
    if eligible_ids&quarantined_ids or eligible_ids|quarantined_ids!=original_ids:
        raise ValueError('Quarantine ID conservation failed')
    return {'eligible':eligible,'quarantined':quarantined,'groups':groups,
            'direct_match_train_ids':sorted(direct)}


def freeze(train_list,test_list,root,out,seed=1337,target=726,family_map=None,no_derivatives_attested=False,
           expected_train=3630,expected_test=840,quarantine_policy_id='disabled',exploratory_unknown_parent_authorized=False):
    if exploratory_unknown_parent_authorized and (family_map or quarantine_policy_id!=QUARANTINE_POLICY):
        raise ValueError('Exploratory unknown-parent mode requires D4 quarantine and no parent map')
    out=Path(out)
    if out.exists():raise FileExistsError(f'Output already exists: {out}')
    train=scan(train_list,root,'train');test=scan(test_list,root,'test')
    if len(train)!=expected_train or len(test)!=expected_test:raise ValueError('Original split counts mismatch')
    evidence=d4_evidence(train,test)
    if quarantine_policy_id not in ('disabled',QUARANTINE_POLICY):raise ValueError('Unknown quarantine policy ID')
    if evidence['train_test_pairs'] and quarantine_policy_id=='disabled':
        out.mkdir(parents=True,exist_ok=False)
        (out/'d4_matches.json').write_text(json.dumps(evidence,indent=2))
        audit={'status':'BLOCKED_CROSS_SPLIT_D4','cross_split_pair_count':len(evidence['train_test_pairs']),
               'd4_exact_check_status':'BLOCKED_CROSS_SPLIT_D4','d4_transform_set':'D4_EXACT_RGB_8_V1',
               'd4_train_test_pair_count':len(evidence['train_test_pairs']),
               'd4_matches_sha256':sha(out/'d4_matches.json'),
               'official_train_list_sha256':sha(train_list),'official_test_list_sha256':sha(test_list),
               'lineage_evidence':'UNKNOWN','scope':evidence['limitations']}
        (out/'split_audit.json').write_text(json.dumps(audit,indent=2))
        raise ValueError(f'Cross-original-split D4 image equivalence found; evidence: {out}')
    families=load_families(family_map,train)
    evidence_bytes=json.dumps(evidence,indent=2).encode()
    partition=partition_training(train,evidence,families,quarantine_policy_id,hashlib.sha256(evidence_bytes).hexdigest())
    eligible=partition['eligible'];quarantined=partition['quarantined'];groups={}
    for row in eligible:groups.setdefault(row['group_id'],[]).append(row)
    actual_target=round(len(eligible)*target/expected_train) if quarantine_policy_id!='disabled' else target
    if not 0<actual_target<len(eligible):raise ValueError('Eligible pool too small for declared validation fraction')
    tr,va=choose_groups(groups,actual_target,seed)
    if not tr or not va:raise ValueError('Empty split')
    coverage=lambda rows:{str(c):sum(c in r['present_classes'] for r in rows) for c in range(4)}
    if not all(coverage(va).values()) or not all(coverage(tr).values()):raise ValueError('Missing declared GT class')
    out.mkdir(parents=True,exist_ok=False)
    (out/'d4_matches.json').write_bytes(evidence_bytes)
    for name,rows in [('dev_train.jsonl',tr),('dev_validation.jsonl',va)]:
        (out/name).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in sorted(rows,key=lambda r:r['sample_id'])))
    if quarantine_policy_id!='disabled':
        (out/'quarantine.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in sorted(quarantined,key=lambda r:r['sample_id'])))
    status='FROZEN' if family_map or (no_derivatives_attested and quarantine_policy_id=='disabled') else 'CANDIDATE_FAMILY_LINEAGE_UNCONFIRMED'
    if exploratory_unknown_parent_authorized:status='FROZEN_EXPLORATORY_D4_ONLY'
    audit={'status':status,'original_train_n':len(train),'original_test_n':len(test),'seed':seed,'method':'group-level deterministic greedy class-appearance stratification',
           'original_shapes':sorted({tuple(r['shape']) for r in train+test}),'target_validation':actual_target,'validation_fraction':target/expected_train,'actual_train':len(tr),'actual_validation':len(va),'groups':len(groups),
           'training_class_appearance':coverage(tr),'validation_class_appearance':coverage(va),
           'training_class_pixels':{str(c):sum(r['class_pixels'][str(c)] for r in tr) for c in range(4)},
           'validation_class_pixels':{str(c):sum(r['class_pixels'][str(c)] for r in va) for c in range(4)},
           'class_semantic_mapping':'UNKNOWN_INTEGER_IDS_ONLY',
           'exploratory_unknown_parent_authorized':exploratory_unknown_parent_authorized,
           'origin_independence':'UNKNOWN',
           'study_scope':'EXPLORATORY_ONLY' if exploratory_unknown_parent_authorized else 'STANDARD_PARENT_GATE',
           'lineage_evidence':'SUPPLIED_PARENT_MAP' if family_map else ('AUTHOR_ATTESTATION_NOT_INDEPENDENT_VERIFICATION' if no_derivatives_attested else 'UNKNOWN'),
           'official_train_list_sha256':sha(train_list),'official_test_list_sha256':sha(test_list),
           'dev_train_sha256':sha(out/'dev_train.jsonl'),'dev_validation_sha256':sha(out/'dev_validation.jsonl'),
           'd4_matches_sha256':sha(out/'d4_matches.json'),
           'd4_exact_check_status':'PASS_CROSS_SPLIT_EXACT_D4_QUARANTINED' if evidence['train_test_pairs'] else 'PASS_NO_CROSS_SPLIT_EXACT_D4',
           'd4_transform_set':'D4_EXACT_RGB_8_V1',
           'd4_train_train_pair_count':len(evidence['train_train_pairs']),
           'd4_train_test_pair_count':len(evidence['train_test_pairs']),
           'family_map_sha256':sha(family_map) if family_map else None,
           'family_map_path':str(Path(family_map).resolve()) if family_map else None,
           'data_governance_policy_id':quarantine_policy_id,
           'original_train_count':len(train),'eligible_train_count':len(eligible),'quarantined_train_count':len(quarantined),
           'original_train_ids_sha256':hashlib.sha256(('\n'.join(sorted(r['sample_id'] for r in train))+'\n').encode()).hexdigest(),
           'eligible_train_ids_sha256':hashlib.sha256(('\n'.join(sorted(r['sample_id'] for r in eligible))+'\n').encode()).hexdigest(),
           'quarantined_train_ids_sha256':hashlib.sha256(('\n'.join(sorted(r['sample_id'] for r in quarantined))+'\n').encode()).hexdigest(),
           'quarantine_manifest_sha256':sha(out/'quarantine.jsonl') if quarantine_policy_id!='disabled' else None,
           'no_derivatives_author_attestation':no_derivatives_attested,
           'scope':evidence['limitations']}
    (out/'split_audit.json').write_text(json.dumps(audit,indent=2));return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['train-list','test-list','data-root','output']:p.add_argument('--'+flag,required=True)
    p.add_argument('--family-map');p.add_argument('--author-attests-no-derived-images',action='store_true')
    p.add_argument('--quarantine-policy-id',default='disabled',choices=['disabled',QUARANTINE_POLICY],help='Candidate only; explicit opt-in, not author approval')
    p.add_argument('--exploratory-unknown-parent-authorized',action='store_true',help='Only after explicit author approval: exact D4 exploration, original parent remains unknown')
    p.add_argument('--seed',type=int,default=1337)
    a=p.parse_args();print(json.dumps(freeze(a.train_list,a.test_list,a.data_root,a.output,a.seed,
        family_map=a.family_map,no_derivatives_attested=a.author_attests_no_derived_images,quarantine_policy_id=a.quarantine_policy_id,
        exploratory_unknown_parent_authorized=a.exploratory_unknown_parent_authorized),indent=2))

if __name__=='__main__':main()

"""Bind existing frozen KSDD2 files to three pending configs; never split data."""
import argparse,copy,json
from pathlib import Path
from .ksdd2_contract import ASSETS,FROZEN_HASHES,config,data_contract,protocol_errors
from .handoff_common import sha256,source_snapshot_sha256
from .ksdd2_support import SEEDS


def prepare(frozen_root,data_root,runs_root,schema_map,output):
    root=Path(frozen_root).resolve();schema=json.loads(Path(schema_map).read_text())
    template=Path(__file__).resolve().parents[1]/'configs/ksdd2_runtime_template_v1.json'
    base=json.loads(template.read_text());base.update(schema=schema,schema_map_sha256=sha256(schema_map),
      data_root=str(Path(data_root).resolve()),runs_root=str(Path(runs_root).resolve()),
      assets={k:str(root/name) for k,name in ASSETS.items()},expected_source_sha256=source_snapshot_sha256())
    base['asset_sha256']={k:sha256(p) for k,p in base['assets'].items()}
    if any(base['asset_sha256'][k]!=h for k,h in FROZEN_HASHES.items()):raise ValueError('Frozen train/validation SHA does not match author identity')
    data_contract(base,check_raw=False)
    configs={}
    for seed in SEEDS:
        c=copy.deepcopy(base);c.update(seed=seed,batch_id=f'ksdd2_v1_s{seed}_{base["asset_sha256"]["train_manifest"][:10]}')
        if protocol_errors(c):raise ValueError('KSDD2 candidate protocol mismatch: '+str(protocol_errors(c)))
        configs[f'ksdd2_seed{seed}.json']=c
    target=Path(output).resolve()
    repo=Path(__file__).resolve().parents[3]
    if target.is_relative_to(repo):raise ValueError('Execution configs must remain outside the source worktree')
    target.mkdir(parents=True,exist_ok=False)
    for name,c in configs.items():
        with (target/name).open('x') as f:json.dump(c,f,indent=2,allow_nan=False);f.write('\n')
    with (target/'SHA256SUMS.txt').open('x') as f:
        for name in configs:f.write(sha256(target/name)+'  '+name+'\n')
    return {name:sha256(target/name) for name in configs}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-root','data-root','runs-root','schema-map','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.frozen_root,a.data_root,a.runs_root,a.schema_map,a.output),indent=2))

if __name__=='__main__':main()

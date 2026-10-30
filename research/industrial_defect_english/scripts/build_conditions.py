"""Freeze class-conditional thresholds from development TRAIN only; never test."""
import argparse,json,hashlib
from pathlib import Path
import torch
from .train_e0 import records,SegmentationData
from ..models.metrics import contour

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-manifest',required=True);p.add_argument('--manifest-sha',required=True)
    p.add_argument('--output',required=True); a=p.parse_args()
    rows=records(a.train_manifest,a.manifest_sha)
    values={c:{'area':[],'complexity':[]} for c in (1,2,3)}
    for _,y in SegmentationData(rows,[200,200]):
        for c in values:
            m=y==c; area=int(m.sum())
            if area:
                values[c]['area'].append(area/y.numel())
                values[c]['complexity'].append(float(contour(m).sum())/area**.5)
    result={'train_manifest_sha256':a.manifest_sha,'input_hw':[200,200],
            'definition':'class-positive GT area fraction and one 3x3 erosion inner contour / sqrt(area)',
            'classes':{c:{k:{'n':len(v),'p33':float(torch.quantile(torch.tensor(v),.33)),
                                  'p67':float(torch.quantile(torch.tensor(v),.67))} if v else None
                          for k,v in fields.items()} for c,fields in values.items()}}
    with Path(a.output).open('x') as f: json.dump(result,f,indent=2)
    print(hashlib.sha256(Path(a.output).read_bytes()).hexdigest())

if __name__=='__main__': main()

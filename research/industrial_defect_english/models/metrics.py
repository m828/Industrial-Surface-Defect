"""CPU validation metrics. Symmetric boundary band is not standard inner BIoU."""
import torch
from torch.nn import functional as F


def dilate(x,t=1):
    return F.max_pool2d(x.float()[None,None],2*t+1,1,t)[0,0].bool()


def erode(x,t=1):
    p=F.pad(x.float()[None,None],(t,t,t,t),value=0)
    return (1-F.max_pool2d(1-p,2*t+1,1,0))[0,0].bool()


def contour(x): return x & ~erode(x)


def boundary_scores(gt,pred,t=2):
    if not gt.any() and not pred.any(): return None,None
    bg=dilate(gt,t)&~erode(gt,t); bp=dilate(pred,t)&~erode(pred,t)
    union=(bg|bp).sum(); biou=float((bg&bp).sum()/union) if union else None
    cg=contour(gt); cp=contour(pred)
    precision=float((cp&dilate(cg,t)).sum()/cp.sum()) if cp.any() else 0.
    recall=float((cg&dilate(cp,t)).sum()/cg.sum()) if cg.any() else 0.
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return biou,f1


def case_metrics(y,pred,n,sample_id):
    rows=[]
    for c in range(1,n):
        gt=y==c; p=pred==c; area=int(gt.sum()); union=int((gt|p).sum())
        row={'sample_id':sample_id,'class_id':c,'gt_area':area,'gt_fraction':area/y.numel(),
             'tp':int((gt&p).sum()),'fp_background':int((p&(y==0)).sum()),
             'fp_other_class':int((p&(y!=0)&~gt).sum()),'fn':int((gt&~p).sum()),
             'iou':float((gt&p).sum()/union) if union else None,
             'inner_boundary_complexity':float(contour(gt).sum())/(area**.5) if area else None}
        tp=row['tp']; fp=row['fp_background']+row['fp_other_class']; fn=row['fn']
        row['precision']=tp/(tp+fp) if tp+fp else None
        row['recall']=tp/(tp+fn) if tp+fn else None
        row['dice']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        for tol in (1,2,3):
            row[f'symmetric_band_iou_{tol}'],row[f'bf1_{tol}']=boundary_scores(gt,p,tol)
        rows.append(row)
    return rows

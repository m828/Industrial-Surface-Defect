"""Device-neutral port of existing loss routes, not a newly proposed loss.

Legacy full/x2/x4 integer-label Laplacian targets and threshold 0.1 retained.
The original loss uses a fresh CUDA Parameter for fixed fusion on every call;
this implementation uses the same constant values on the current device.
"""
import torch
from torch.nn import functional as F


def boundary_target(labels):
    x=labels.unsqueeze(1).float()
    kernel=x.new_tensor([-1,-1,-1,-1,8,-1,-1,-1,-1]).view(1,1,3,3)
    bands=[]
    for stride in (1,2,4):
        z=F.conv2d(x,kernel,stride=stride,padding=1).clamp_min(0)
        z=F.interpolate(z,labels.shape[-2:],mode='nearest')
        bands.append((z>0.1).float())
    return ((.6*bands[0]+.3*bands[1]+.1*bands[2])>0.1).float()


def detail_loss(logits, labels):
    target=boundary_target(labels)
    p=logits.sigmoid().flatten(1); t=target.flatten(1)
    dice=(1-(2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1)).mean()
    return F.binary_cross_entropy_with_logits(logits,target)+dice


def ohem(logits, labels, threshold=.7, min_kept=100000):
    losses=F.cross_entropy(logits,labels,reduction='none').flatten()
    probs=logits.softmax(1).gather(1,labels.unsqueeze(1)).flatten()
    sorted_p,indices=probs.sort()
    cutoff=max(threshold,float(sorted_p[min(min_kept,len(sorted_p)-1)].detach()))
    keep=sorted_p<cutoff
    # This edge case was undefined (NaN) in legacy code; explicit new-run fallback.
    if not keep.any(): return losses.mean()
    return losses[indices][keep].mean()


def joint_loss(outputs, labels, weights=(10.,1.,3.,3.)):
    if len(outputs)==3 and len(weights)==4: weights=weights[:3]
    if len(weights)!=len(outputs): raise ValueError('Loss count / weight count mismatch')
    if labels.min()<0 or labels.max()>=outputs[0].shape[1]:
        raise ValueError('E0 expects valid integer labels, no ignore pixels')
    if len(outputs) not in (3,4): raise ValueError('Expected three seg routes and optional detail')
    target=F.one_hot(labels,outputs[0].shape[1]).permute(0,3,1,2).float()
    components=[ohem(outputs[0],labels),F.binary_cross_entropy_with_logits(outputs[1],target),
                ohem(outputs[2],labels)]
    if len(outputs)==4: components.append(detail_loss(outputs[3],labels))
    return sum(w*l for w,l in zip(weights,components)),components


# Protocol IDs are intentionally explicit: selecting the candidate is an author
# decision, not a silent change to the pre-existing joint_loss() default.
LOSS_PROTOCOL_CURRENT = 'e0_multiclass_detail_v1'
LOSS_PROTOCOL_HISTORICAL = 'neu_a2ms_mask_aware_candidate_v1'
LOSS_PROTOCOL_IDS = (LOSS_PROTOCOL_CURRENT, LOSS_PROTOCOL_HISTORICAL)


def _historical_ohem(logits, labels):
    """Original OHEM for valid E0 labels, including its empty-selection NaN."""
    pixel_losses = F.cross_entropy(logits, labels, reduction='none').contiguous().view(-1)
    probs = F.softmax(logits, dim=1).gather(1, labels.unsqueeze(1))
    probs, indices = probs.contiguous().view(-1).sort()
    cutoff = max(probs[min(100000, probs.numel() - 1)], 0.7)
    return pixel_losses[indices][probs < cutoff].mean()


def _historical_bce(logits, labels):
    # The historic routine constructs a pre-one-hot mask and divides by its
    # mean + 1e-10. E0 explicitly rejects ignore labels rather than guessing.
    target = F.one_hot(labels, logits.shape[1]).permute(0, 3, 1, 2).float()
    mask = (labels.unsqueeze(1) != 255).float()
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (loss * mask).mean() / (mask.mean() + 1e-10)


def _historical_detail_loss(logits, labels):
    # The original creates fresh float32 CUDA kernels per invocation. This
    # device-neutral equivalent keeps the original conv/threshold operations;
    # no claim is made of bitwise CPU/GPU equality.
    x = labels.unsqueeze(1).to(dtype=torch.float32)
    kernel = x.new_tensor([-1,-1,-1,-1,8,-1,-1,-1,-1]).view(1,1,3,3)
    bands = []
    for stride in (1, 2, 4):
        z = F.conv2d(x, kernel, stride=stride, padding=1).clamp_min(0)
        if stride != 1:
            z = F.interpolate(z, x.shape[-2:], mode='nearest')
        bands.append((z > 0.1).to(dtype=torch.float32))
    pyramid = torch.cat(bands, dim=1)
    fuse = x.new_tensor([0.6, 0.3, 0.1]).view(1,3,1,1)
    target = (F.conv2d(pyramid, fuse) > 0.1).to(dtype=torch.float32)
    # Preserve original width-only resize check and its align_corners setting.
    if logits.shape[-1] != target.shape[-1]:
        logits = F.interpolate(logits, target.shape[-2:], mode='bilinear', align_corners=True)
    p = logits.sigmoid().flatten(1)
    t = target.flatten(1)
    dice = (1 - (2 * (p*t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()
    return F.binary_cross_entropy_with_logits(logits, target) + dice


def loss_component_names(loss_protocol_id):
    if loss_protocol_id == LOSS_PROTOCOL_CURRENT:
        return ('seg1_multiclass', 'seg2_multiclass', 'seg3_multiclass', 'detail')
    if loss_protocol_id == LOSS_PROTOCOL_HISTORICAL:
        return ('seg1_multiclass', 'seg2_multiclass', 'seg3_multiclass', 'detail',
                'seg1_mask_aware', 'seg2_mask_aware', 'seg3_mask_aware')
    raise ValueError('Unknown loss_protocol_id: ' + str(loss_protocol_id))


def joint_loss_with_protocol(outputs, labels, weights, loss_protocol_id):
    """Select a loss explicitly; candidate is for four-output E0 models only.

    Returned components are unweighted and ordered as loss_component_names().
    A historic empty-OHEM selection yields NaN; the trainer must stop on it.
    """
    loss_component_names(loss_protocol_id)
    if loss_protocol_id == LOSS_PROTOCOL_CURRENT:
        return joint_loss(outputs, labels, weights)
    if len(outputs) != 4 or len(weights) != 4:
        raise ValueError('Historical candidate requires three segmentation heads, one detail head and four weights')
    if labels.dtype != torch.long or labels.min() < 0 or labels.max() >= outputs[0].shape[1]:
        raise ValueError('E0 expects valid int64 labels with no ignore pixels')
    binary = (labels != 0).long()
    routes = (_historical_ohem, _historical_bce, _historical_ohem)
    multiclass = [fn(output, labels) for fn, output in zip(routes, outputs[:3])]
    detail = _historical_detail_loss(outputs[3], labels)
    binary_logits = [torch.stack((o[:, 0], o[:, 1:].sum(dim=1)), dim=1) for o in outputs[:3]]
    mask_aware = [fn(output, binary) for fn, output in zip(routes, binary_logits)]
    components = [*multiclass, detail, *mask_aware]
    # Legacy accumulates binary branch and multiclass branch separately.
    seg_loss = sum(w * l for w, l in zip(weights, [*multiclass, detail]))
    mask_loss = sum(w * l for w, l in zip(weights[:3], mask_aware))
    return mask_loss + seg_loss, components

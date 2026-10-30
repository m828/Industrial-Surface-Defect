"""CPU-only synthetic checks of the optional historical loss candidate."""
import unittest

import torch
from torch.nn import functional as F

from research.industrial_defect_english.models.losses import (
    LOSS_PROTOCOL_CURRENT, LOSS_PROTOCOL_HISTORICAL, _historical_detail_loss,
    joint_loss, joint_loss_with_protocol, loss_component_names,
)
from research.industrial_defect_english.vendor.loss import (
    ohem_cross_entropy, bce_with_logits_loss,
)


def reference_detail(logits, labels):
    # A literal device-neutral translation of the archived CUDA-only routine.
    k = torch.tensor([-1,-1,-1,-1,8,-1,-1,-1,-1], dtype=torch.float32).reshape(1,1,3,3)
    x = labels.unsqueeze(1).float()
    edges = []
    for stride in (1,2,4,8):
        e = F.conv2d(x, k, stride=stride, padding=1).clamp(min=0)
        if stride != 1:
            e = F.interpolate(e, x.shape[2:], mode='nearest')
        e[e > .1] = 1
        e[e <= .1] = 0
        edges.append(e)
    targets = torch.stack(edges[:3], dim=1).squeeze(2)
    targets = F.conv2d(targets, torch.tensor([[.6],[.3],[.1]]).reshape(1,3,1,1))
    targets[targets > .1] = 1
    targets[targets <= .1] = 0
    if logits.shape[-1] != targets.shape[-1]:
        logits = F.interpolate(logits, targets.shape[2:], mode='bilinear', align_corners=True)
    p = logits.sigmoid().view(logits.shape[0], -1)
    t = targets.view(targets.shape[0], -1)
    dice = (1-(2*(p*t).sum(1)+1)/(p.sum(1)+t.sum(1)+1)).mean()
    return F.binary_cross_entropy_with_logits(logits, targets)+dice


class HistoricLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.labels = torch.randint(0, 4, (2, 9, 11))
        self.out = [torch.randn(2, 4, 9, 11, requires_grad=True) for _ in range(3)]
        self.out.append(torch.randn(2, 1, 9, 11, requires_grad=True))
        self.weights = [10,1,3,3]

    def test_reference_each_route_and_weighted_total(self):
        total, parts = joint_loss_with_protocol(self.out, self.labels, self.weights, LOSS_PROTOCOL_HISTORICAL)
        routes = (ohem_cross_entropy, bce_with_logits_loss, ohem_cross_entropy)
        seg = [fn(o, self.labels) for fn,o in zip(routes,self.out[:3])]
        binary = (self.labels != 0).long()
        bin_logits = [torch.stack((o[:,0],o[:,1:].sum(1)),dim=1) for o in self.out[:3]]
        aware = [fn(o,binary) for fn,o in zip(routes,bin_logits)]
        expected = [*seg,reference_detail(self.out[3],self.labels),*aware]
        self.assertEqual(len(parts),7)
        for actual, ref in zip(parts,expected):
            torch.testing.assert_close(actual,ref,rtol=0,atol=0)
        ref_total = sum(w*l for w,l in zip(self.weights[:3],aware)) + sum(w*l for w,l in zip(self.weights,expected[:4]))
        torch.testing.assert_close(total,ref_total,rtol=0,atol=0)
        total.backward()
        for tensor in self.out:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_detail_resize_matches_reference(self):
        logits=torch.randn(2,1,5,6)
        torch.testing.assert_close(_historical_detail_loss(logits,self.labels),reference_detail(logits,self.labels),rtol=0,atol=0)

    def test_historical_empty_ohem_remains_nan(self):
        labels=torch.zeros((1,4,4),dtype=torch.long)
        logits=torch.full((1,4,4,4),-1000.)
        logits[:,0]=1000.
        outs=[logits.clone() for _ in range(3)]+[torch.zeros(1,1,4,4)]
        total,parts=joint_loss_with_protocol(outs,labels,self.weights,LOSS_PROTOCOL_HISTORICAL)
        self.assertTrue(torch.isnan(parts[0]))
        self.assertTrue(torch.isnan(total))
        self.assertTrue(torch.isfinite(joint_loss(outs,labels,self.weights)[0]))

    def test_current_default_unchanged_and_invalid_protocol_rejected(self):
        old=joint_loss(self.out,self.labels,self.weights)
        new=joint_loss_with_protocol(self.out,self.labels,self.weights,LOSS_PROTOCOL_CURRENT)
        torch.testing.assert_close(old[0],new[0],rtol=0,atol=0)
        self.assertEqual(len(loss_component_names(LOSS_PROTOCOL_HISTORICAL)),7)
        with self.assertRaises(ValueError):
            joint_loss_with_protocol(self.out,self.labels,self.weights,'unknown')
        with self.assertRaises(ValueError):
            joint_loss_with_protocol(self.out,self.labels.float(),self.weights,LOSS_PROTOCOL_HISTORICAL)


if __name__ == '__main__':
    unittest.main()

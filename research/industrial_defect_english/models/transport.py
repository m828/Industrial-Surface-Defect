"""Experimental, supervision-coupled feature exchange; no performance claim.

The sum identity is algebraic tensor conservation only. Feature channels are not
assumed semantically aligned. This assumption is a specific scientific risk.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .factory import FORMAL_A2


class RelationTransport(nn.Module):
    def __init__(self, bound=0.25):
        super().__init__()
        self.bound = bound
        self.predict = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1, padding_mode='replicate'),
                                     nn.ReLU(), nn.Conv2d(8, 1, 1))
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self, detail, semantic, global_condition=False):
        # Descriptor invariant to joint positive rescaling, finite at zero.
        dl = detail.square().mean(1, keepdim=True).add(1e-6).sqrt()
        sr = semantic.square().mean(1, keepdim=True).add(1e-6).sqrt()
        cosine = (detail * semantic).mean(1, keepdim=True) / (dl * sr)
        diff = (detail-semantic).abs().mean(1, keepdim=True) / (dl+sr)
        q = torch.cat([cosine, diff, dl/(dl+sr), sr/(dl+sr)], 1)
        if global_condition:
            q = q.mean((2, 3), keepdim=True).expand_as(q)
        beta = self.bound * torch.tanh(self.predict(q))
        return beta * (semantic-detail), beta


class CoupledTransportModel(FORMAL_A2.DSMONet):
    def __init__(self, num_classes, backbone, variant='coupled_transport'):
        super().__init__(num_classes, backbone)
        self.variant = variant
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(173)
            self.transport = RelationTransport()

    def forward(self, x):
        size = x.shape[-2:]
        features = self.backbone(x)
        features = [features[i] for i in self.backbone_indices]
        s = self.arm1(features[-1], self.cm(features[-1]))
        body, edge = self.squeeze_body_edge(s)
        e = self.conv_up(edge)
        l = F.interpolate(self.laplacian(self.bot_fine(features[0])),
                          e.shape[-2:], mode='bilinear', align_corners=False)
        a = self.se(s)
        r = self.conv_up(a)
        if self.variant == 'head_shift_only' and not self.training:
            t, beta = torch.zeros_like(l), None
        else:
            t, beta = self.transport(l, r, global_condition=self.variant == 'capacity_control')
        if self.variant == 'head_shift_only':
            # No transport in either fusion: tests a training-only head change.
            d = self.edge_fusion(torch.cat([e, l, r], 1))
            m = self.conv_up(s+body+a)
            head_input = l+t
        else:
            l1 = l+t
            coupled = self.variant not in ('capacity_control', 'uncoupled_transport')
            r1 = r-t if coupled else r
            d = self.edge_fusion(torch.cat([e, l1, r1], 1))
            m = self.conv_up(s+body+a)
            if coupled and self.variant != 'stage1_only':
                if self.variant == 'recomputed_final_transport':
                    t_final, _ = self.transport(d, m)
                    m = m-t_final
                else:
                    m = m-t
            head_input = l if self.variant in ('capacity_control', 'no_head_shift') else l1
        out = self.arm2(d, m)
        if self.training:
            logits = [self.seg_heads(out), self.seg_heads(d), self.seg_heads(s),
                      self.seg_head_detailloss(head_input)]
        else:
            logits = [self.seg_heads(out)]
        return [F.interpolate(v, size, mode='bilinear', align_corners=False) for v in logits]

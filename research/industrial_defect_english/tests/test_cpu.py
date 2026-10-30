"""Only random CPU tensors; no dataset/model performance evaluation."""
import copy
import json
from pathlib import Path
import unittest
import torch
from ..models import build_model, load_compatible
from ..models.transport import RelationTransport
from ..models.losses import detail_loss, joint_loss, boundary_target

torch.set_num_threads(4)


class CPUChecks(unittest.TestCase):
    def test_factory_init_and_state_compatibility(self):
        old=build_model('formal_a2ms').eval(); new=build_model('coupled_transport').eval()
        report=load_compatible(new,old.state_dict(),allow_new=True)
        self.assertEqual(len(report['missing']),4)
        for k,v in old.state_dict().items(): self.assertTrue(torch.equal(v,new.state_dict()[k]),k)
        with self.assertRaises(RuntimeError): load_compatible(new,{'bad':torch.tensor(1.)})
        x=torch.randn(1,3,200,200)
        with torch.no_grad(): a=old(x); b=new(x)
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)

    def test_transport_identity_sum_and_update(self):
        torch.manual_seed(3)
        module=RelationTransport()
        l=torch.randn(2,128,20,20,requires_grad=True)
        r=torch.randn_like(l,requires_grad=True)
        t,b=module(l,r)
        self.assertEqual(float(t.detach().abs().max()),0.)
        opt=torch.optim.SGD(module.parameters(),lr=.1)
        ((l+t-r).square().mean()).backward()
        self.assertGreater(float(module.predict[-1].weight.grad.abs().sum()),0)
        self.assertEqual(float(module.predict[0].weight.grad.abs().sum()),0)
        opt.step(); opt.zero_grad(set_to_none=True)
        t,b=module(l,r)
        torch.testing.assert_close((l+t)+(r-t),l+r,atol=1e-6,rtol=1e-6)
        self.assertLessEqual(float(b.detach().abs().max()),.25)
        (l+t-r).square().mean().backward()
        self.assertGreater(float(module.predict[0].weight.grad.abs().sum()),0)
        z=torch.zeros(2,128,9,11)
        self.assertTrue(torch.isfinite(module(z,z)[0]).all())

    def test_train_output_sharing_and_loss_gradients(self):
        for variant,n in [('formal_base',3),('formal_a2ms',4),('coupled_transport',4),('capacity_control',4)]:
            model=build_model(variant).train()
            calls=[]; heads=[]; raw_l=[]; ts=[]; extra=[]
            hook=model.seg_heads.register_forward_hook(lambda m,a,o: calls.append(id(m)))
            if n==4:
                h=model.seg_head_detailloss.register_forward_pre_hook(lambda m,a: heads.append(a[0]))
            if n==4:
                extra.append(model.laplacian.register_forward_hook(lambda m,a,o: raw_l.append(o)))
                if variant=='coupled_transport':
                    extra.append(model.transport.register_forward_hook(lambda m,a,o: ts.append(o[0])))
            out=model(torch.randn(2,3,200,200))
            self.assertEqual(len(out),n); self.assertEqual(len(calls),3)
            self.assertEqual(len(set(calls)),1)
            self.assertEqual(int(model.seg_heads.conv._batch_norm.num_batches_tracked),3)
            if n==4:
                expected=torch.nn.functional.interpolate(raw_l[0],heads[0].shape[-2:],mode='bilinear',align_corners=False)
                if variant=='coupled_transport': expected=expected+ts[0]
                torch.testing.assert_close(heads[0],expected,rtol=0,atol=0)
            self.assertEqual([list(x.shape) for x in out],[[2,4,200,200]]*3+([[2,1,200,200]] if n==4 else []))
            self.assertTrue(all(torch.isfinite(x).all() for x in out))
            y=torch.randint(0,4,(2,200,200))
            loss,parts=joint_loss(out,y); self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name in ['arm2','se','edge_fusion','laplacian']:
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,name).parameters()),name)
            if variant=='coupled_transport':
                self.assertGreater(float(model.transport.predict[-1].weight.grad.abs().sum()),0)
            hook.remove()
            for e in extra: e.remove()
            if n==4: h.remove()
            del model, out,loss

    def test_detail_supervision_reaches_semantics_after_update(self):
        model=build_model('coupled_transport').train()
        opt=torch.optim.SGD(model.parameters(),lr=1e-3)
        x=torch.randn(2,3,200,200); y=torch.randint(0,4,(2,200,200))
        detail_loss(model(x)[3],y).backward()
        before=model.transport.predict[-1].weight.detach().clone()
        opt.step(); opt.zero_grad(set_to_none=True)
        self.assertFalse(torch.equal(before,model.transport.predict[-1].weight))
        detail_loss(model(x)[3],y).backward()
        self.assertIsNotNone(model.se.fc.weight.grad)
        self.assertGreater(float(model.se.fc.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.transport.predict[0].weight.grad.abs().sum()),0)
        # Original detail input is independent of se; no detail-only gradient.
        old=build_model('formal_a2ms').train()
        detail_loss(old(torch.randn(2,3,200,200))[3],torch.randint(0,4,(2,200,200))).backward()
        self.assertIsNone(old.se.fc.weight.grad)

    def test_active_correction_insertions(self):
        model=build_model('coupled_transport').train()
        with torch.no_grad(): model.transport.predict[-1].bias.fill_(.2)
        seen={'up':[],'lap':[],'t':[],'fusion':[],'arm':[],'head':[]}
        handles=[model.conv_up.register_forward_hook(lambda m,a,o:seen['up'].append(o)),
                 model.laplacian.register_forward_hook(lambda m,a,o:seen['lap'].append(o)),
                 model.transport.register_forward_hook(lambda m,a,o:seen['t'].append(o[0])),
                 model.edge_fusion.register_forward_pre_hook(lambda m,a:seen['fusion'].append(a[0])),
                 model.arm2.register_forward_pre_hook(lambda m,a:seen['arm'].append(a)),
                 model.seg_head_detailloss.register_forward_pre_hook(lambda m,a:seen['head'].append(a[0]))]
        with torch.no_grad(): model(torch.randn(2,3,200,200))
        e,r,m=seen['up']; t=seen['t'][0]
        l=torch.nn.functional.interpolate(seen['lap'][0],e.shape[-2:],mode='bilinear',align_corners=False)
        self.assertGreater(float(t.abs().sum()),0)
        torch.testing.assert_close(seen['fusion'][0],torch.cat([e,l+t,r-t],1),rtol=0,atol=0)
        torch.testing.assert_close(seen['arm'][0][1],m-t,rtol=0,atol=0)
        torch.testing.assert_close(seen['head'][0],l+t,rtol=0,atol=0)
        for h in handles:h.remove()

    def test_nonzero_transport_inference_and_sizes(self):
        model=build_model('coupled_transport').eval()
        with torch.no_grad(): model.transport.predict[-1].bias.fill_(.2)
        calls=[]; head=model.seg_head_detailloss.register_forward_hook(lambda *a: calls.append(1))
        for h,w in [(200,200),(224,256)]:
            with torch.no_grad(): out=model(torch.randn(1,3,h,w))
            self.assertEqual(len(out),1); self.assertEqual(list(out[0].shape),[1,4,h,w])
            self.assertTrue(torch.isfinite(out[0]).all())
        self.assertEqual(calls,[]); head.remove()

    def test_control_parameter_counts_and_detail_target(self):
        old=build_model('formal_a2ms'); new=build_model('coupled_transport'); control=build_model('capacity_control')
        count=lambda m:sum(p.numel() for p in m.parameters())
        self.assertEqual(count(new)-count(old),305); self.assertEqual(count(new),count(control))
        y=torch.zeros(2,20,20,dtype=torch.long); y[:,4:9,6:11]=1
        target=boundary_target(y)
        self.assertEqual(set(target.unique().tolist()),{0.,1.})
        self.assertEqual(list(target.shape),[2,1,20,20])

if __name__=='__main__': unittest.main()

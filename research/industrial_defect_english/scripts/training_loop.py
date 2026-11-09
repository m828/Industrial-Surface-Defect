"""One shared update loop; legacy NEU branch retained, explicit task hooks optional."""
import json,os,random,time
import numpy as np
import torch
from ..models.losses import joint_loss_with_protocol,loss_component_names
from .e0_contract import check_budget,require_finite,require_finite_loss,response_stats,summarize_validation


def execute_updates(model,optimizer,train_loader,val_loader,cfg,run,metadata,started,runtime_budget,
                    sample_rng,val_rng,augmentation_rng,iterator,legacy_validate=None,task=None,device='cuda',conditions=None):
    it=iterator
    training_response=[];response_hook=None;capture_training_response=False
    if (task is not None and hasattr(model,'transport')) or cfg.get('experiment_stage') in ('E1-B','E1-C'):
        def training_diagnostic(module,inputs,outputs):
            if capture_training_response and module.training:
                with torch.no_grad():training_response.append(response_stats(inputs[0],inputs[1],outputs[0],outputs[1]))
        response_hook=model.transport.register_forward_hook(training_diagnostic)
    with (run/'metrics.jsonl').open('x') as log, (run/'updates.jsonl').open('x') as updates:
        for step in range(1,cfg['optimizer_updates']+1):
            check_budget(started,run,runtime_budget,check_disk=step%100==1)
            step_started=time.perf_counter()
            try: batch=next(it)
            except StopIteration: it=iter(train_loader); batch=next(it)
            if task is None:x,y=batch
            model.train(); optimizer.zero_grad(set_to_none=True)
            capture_training_response=(step%cfg['validation_every']==0 or step==cfg['optimizer_updates'])
            training_response.clear()
            if task is None:
                total,parts=joint_loss_with_protocol(model(x.cuda()),y.cuda(),cfg['loss_weights'],cfg['loss_protocol_id'])
                names=loss_component_names(cfg['loss_protocol_id']);diagnostic=None
            else:
                total,parts,diagnostic=task.training_loss(model,batch,device)
                names=task.component_names
            require_finite_loss(total,parts,names,run,step,metadata['completed_updates'],cfg['loss_protocol_id'])
            total.backward()
            require_finite(((n,p.grad) for n,p in model.named_parameters()),'gradient')
            optimizer.step()
            metadata['completed_updates']=step
            updates.write(json.dumps({'completed_updates':step})+'\n'); updates.flush()
            require_finite(model.named_parameters(),'parameter')
            if str(device).startswith('cuda'):torch.cuda.synchronize()
            step_seconds=time.perf_counter()-step_started
            row={'optimizer_updates':step,'step_seconds':step_seconds,'elapsed_wall_seconds':time.perf_counter()-started,'loss':float(total.detach()),'components':[float(v.detach()) for v in parts],'lr':optimizer.param_groups[0]['lr']}
            if (task is not None or cfg.get('experiment_stage') in ('E1-B','E1-C')) and capture_training_response:
                row['training_transport_response']=list(training_response)
                row['mechanism_response_scope']=('KSDD2 feature grid diagnostics include padding dependence' if task is not None else ('TRAIN_LAST_BATCH_FEATURE_DIAGNOSTIC; two ordered calls: stage1, final' if cfg['experiment_stage']=='E1-C' else 'TRAIN_LAST_BATCH_FEATURE_DIAGNOSTIC; head_shift_only has no inference transport'))
            if diagnostic is not None:
                row['loss_diagnostics']=diagnostic
                counts=metadata.setdefault('ohem_fallback_counts',{'0':0,'2':0})
                for route in diagnostic['empty_hard_pixel_ce_fallback_routes']:counts[str(route)]+=1
            if step%cfg['validation_every']==0 or step==cfg['optimizer_updates']:
                row['validation']=(task.validate(model,val_loader,lambda:check_budget(started,run,runtime_budget,False),device) if task is not None else legacy_validate(model,val_loader,cfg['num_classes'],conditions,lambda:check_budget(started,run,runtime_budget,False)))
                row['validation_summary']=(task.validation_summary(row['validation']) if task is not None else summarize_validation(row['validation']))
                row['stop_flags']=[]  # Safety-only: performance and response remain diagnostics.
                check_budget(started,run,runtime_budget)
                if hasattr(model,'transport'):
                    row['transport_last_weight_norm']=float(model.transport.predict[-1].weight.detach().norm())
                recovery={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'optimizer_updates':step,'config':cfg,
                          'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                          'cuda_rng':torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],'sampler_rng':sample_rng.get_state(),
                          'validation_rng':val_rng.get_state(),'augmentation_rng':augmentation_rng.getstate()}
                torch.save(recovery,run/'recovery.tmp')
                os.replace(run/'recovery.tmp',run/'recovery.pt')
            log.write(json.dumps(row)+'\n'); log.flush()
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'optimizer_updates':step,'config':cfg},run/'endpoint.pt')
    check_budget(started,run,runtime_budget)
    if response_hook is not None:response_hook.remove()

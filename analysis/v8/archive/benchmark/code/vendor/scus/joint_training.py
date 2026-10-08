"""Full-input joint training with label-free internal-validation selection."""
from __future__ import annotations
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .artifacts import sha256
from .runtime import write_json, stage_directory, cosine_distance
from .joint_reconstruction import JointModel, AllGeneCache, biological_folds, backward_cell, mask_tokens, reconstruction, load_joint_data


def save_state(path, model, optimizer, discriminator_optimizer, epoch, identity):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    discriminator_optimizer=discriminator_optimizer.state_dict(), epoch=epoch,
                    identity=identity, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    checkpoint_eligible=False, purpose=identity['config']['align'].get('execution','pilot')
                    if 'config' in identity else 'test'), temporary)
    os.replace(temporary, path)


def validate_resume_extension(old, new, checkpoint_epoch):
    """Explicitly extend a completed run; only epoch budget, location and device may change."""
    if new['config']['align'].get('resume_extension') is not True:
        raise ValueError('Epoch extension requires explicit resume_extension: true')
    old_count = old['config']['align']['epochs']
    new_count = new['config']['align']['epochs']
    if not isinstance(new_count, int) or new_count <= old_count or checkpoint_epoch != old_count - 1:
        raise ValueError('Extension must start at the completed old epoch budget and increase epochs')
    def fixed(identity):
        result = json.loads(json.dumps(identity))
        cfg = result['config']
        for key in ['tag', 'output_dir', 'device']:
            cfg.pop(key, None)
        for key in ['epochs', 'resume_extension']:
            cfg['align'].pop(key, None)
        # This module implements the audited extension; model/batch/QC code must remain exact.
        result.get('implementation_sha256', {}).pop('joint_training.py', None)
        return result
    if fixed(old) != fixed(new):
        raise ValueError('Extension scientific config, input or implementation mismatch')


def restore_cuda_rng(states, old_identity, new_identity):
    """Transfer the active training generator when the physical CUDA device changes."""
    if states is None:
        return
    old_device = torch.device(old_identity.get('config', {}).get('device', 'cpu'))
    new_device = torch.device(new_identity.get('config', {}).get('device', 'cpu'))
    if old_device != new_device:
        if old_device.type != 'cuda' or new_device.type != 'cuda' or old_device.index is None or new_device.index is None:
            raise ValueError('Device migration requires explicit physical CUDA indices')
        if old_device.index >= len(states):
            raise ValueError('Checkpoint lacks source-device CUDA RNG state')
        torch.cuda.set_rng_state(states[old_device.index], device=new_device)
    else:
        torch.cuda.set_rng_state_all(states)


def restore_state(path, model, optimizer, discriminator_optimizer, identity, migration=False, extension=False):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['identity'] != identity:
        if extension:
            validate_resume_extension(payload['identity'], identity, int(payload['epoch']))
        elif migration:
            validate_execution_migration(payload['identity'],identity)
        else:
            raise ValueError('Resume input/config/split hash mismatch')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    discriminator_optimizer.load_state_dict(payload['discriminator_optimizer'])
    random.setstate(payload['python_rng']); np.random.set_state(payload['numpy_rng'])
    torch.set_rng_state(payload['torch_rng'])
    restore_cuda_rng(payload['cuda_rng'], payload['identity'], identity)
    return int(payload['epoch']) + 1


def validate_execution_migration(old,new):
    """Allow only declared batching execution changes; scientific identity is exact."""
    for key in ['checkpoint_sha256','vocab_sha256','manifest_sha256','token_source_sha256']:
        if old.get(key)!=new.get(key):raise ValueError(f'Migration input mismatch: {key}')
    if old['implementation_sha256']['joint_reconstruction.py']!=new['implementation_sha256']['joint_reconstruction.py']:
        raise ValueError('Migration cannot change encoder, masking or reconstruction primitives')
    def scientific(identity):
        cfg=json.loads(json.dumps(identity['config']))
        for k in ['tag','output_dir']:cfg.pop(k,None)
        for k in ['microbatch_size','effective_batch_cells','activation_checkpointing','discriminator_update',
                  'microbatch_candidates','memory_budget_gib','resume_execution_migration','eval_batch_size','batch_backend']:
            cfg['align'].pop(k,None)
        return cfg
    if scientific(old)!=scientific(new):raise ValueError('Migration scientific config mismatch')
    ac=new['config']['align']
    if ac.get('effective_batch_cells',32)!=32 or ac.get('discriminator_update','per_cell')!='per_cell':
        raise ValueError('Migration changes update schedule')


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def update_loss_selection(out, ac, epoch):
    if ac.get('selection_policy') != 'weighted_train_postwarmup' or epoch < 4:
        return
    frame = pd.read_csv(out/f'epoch_{epoch:03d}_train.csv')
    adv = float(frame.loss_adv.mean()) if 'loss_adv' in frame else 0.0
    rec = float(frame.loss_rec.mean()) if 'loss_rec' in frame else 0.0
    adv_weight = float(ac.get('adv_weight', .1)); rec_weight = float(ac.get('rec_weight', .3))
    score = adv_weight*adv + rec_weight*rec
    path = out/'loss_selection.json'
    prior = json.loads(path.read_text()) if path.exists() else None
    if prior is None or (score, epoch) < (prior['weighted_train_loss'], prior['selected_epoch']):
        write_json(path, dict(selected_epoch=epoch, selected_checkpoint=f'epoch_{epoch:03d}.pt',
            weighted_train_loss=float(score), loss_adv=adv if adv_weight else None,
            loss_rec=rec if rec_weight else None, adv_weight=adv_weight, rec_weight=rec_weight,
            policy='weighted_train_postwarmup',
            qc_used_as_gate=False, not_validation_total_loss=True))


@torch.no_grad()
def evaluate(model, data, indices, device, seed, out, label):
    """All internal-validation cells, two masks fixed across checkpoints."""
    if getattr(model,'evaluation_batch_size',1)>1:
        from .joint_batch import evaluate_batched
        return evaluate_batched(model,data,indices,device,seed,out,label,model.evaluation_batch_size)
    model.eval(); rows=[]; states=[]
    for index in indices:
        b=to_device(data[index],device); g,v,s=(b[k] for k in ['gene_ids','value_bins','splice_flags'])
        paired=v[0,::2].gt(0)&v[0,1::2].gt(0)
        if not paired.any(): raise ValueError('No valid paired genes in validation cell')
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            h,z=model.hidden(g,v,s)
        state=((z[0,::2][paired].float()+z[0,1::2][paired].float())/2).mean(0)
        states.append(state.cpu().numpy())
        distances=cosine_distance(z[0,::2][paired],z[0,1::2][paired])
        if not torch.isfinite(distances).all() or (distances < -1e-6).any() or (distances > 2+1e-6).any():
            raise FloatingPointError('Invalid validation distances')
        rec=[]
        for replica in range(2):
            mask=mask_tokens(v,model.encoder.mask_ratio,seed,0,b['cell_id'],replica)
            masked=v.clone(); masked[mask]=model.encoder.mask_token
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                _,mz=model.hidden(g,masked,s)
                metrics=reconstruction(model.encoder.head_gene(mz),v,s,mask)
            rec.append({k:float(x) for k,x in metrics.items()})
        row=dict(cell_id=b['cell_id'],genes=g.numel()//2,paired_genes=int(paired.sum()),
                 distance_mean=float(distances.mean()),distance_median=float(distances.median()),
                 distance_iqr=float(torch.quantile(distances,.75)-torch.quantile(distances,.25)))
        row.update({k:np.mean([r[k] for r in rec]) for k in rec[0]})
        rows.append(row)
    frame=pd.DataFrame(rows)
    if not np.isfinite(frame.select_dtypes('number').to_numpy()).all(): raise FloatingPointError('Nonfinite validation metrics')
    frame.to_csv(out/f'{label}_validation_cells.csv',index=False)
    np.save(out/f'{label}_validation_state.npy',np.stack(states))
    return {k:float(frame[k].mean()) for k in frame.select_dtypes('number').columns}


def fit(cfg, tag=None, device_name=None, epochs=None, steps_per_epoch=None):
    ac=cfg['align']
    microbatch=ac.get('microbatch_size',1)
    if not isinstance(microbatch,int) or microbatch not in [1,2,4,8,16,32]:raise ValueError('Resolve microbatch_size before fit')
    if ac.get('effective_batch_cells',32)!=32 or ac.get('discriminator_update','per_cell')!='per_cell':
        raise ValueError('Requires 32-cell groups and per-cell discriminator updates')
    formal=ac.get('execution')=='formal'
    if ac.get('execution') not in ('pilot','formal'):
        raise ValueError('Specify execution: pilot or formal')
    if ac.get('gene_selection')!='all_eligible' or ac.get('max_genes') is not None:
        raise ValueError('Joint mode requires all_eligible and max_genes: null')
    if steps_per_epoch is not None: raise ValueError('Joint epochs must traverse every training cell')
    seed=int(cfg.get('seed',42)); count=int(epochs or ac.get('epochs',1))
    if count<1: raise ValueError('epochs must be positive')
    device=torch.device(device_name or cfg.get('device','cpu'))
    out=stage_directory(cfg,'align-fit',tag)
    # Include effective CLI overrides in the immutable resume identity.
    cfg=json.loads(json.dumps(cfg)); cfg['align']['epochs']=count; cfg['device']=str(device)
    lock=out/'run.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try:
        write_json(out/'status.json',dict(status='preparing',pid=os.getpid(),checkpoint_eligible=False))
        data=load_joint_data(cfg)
        folds=biological_folds(data.cells,int(ac.get('split_seed',42)))
        fold=folds[int(ac.get('fold',0))]
        manifest=[]
        for split in ['train','validation','test']:
            for index in fold[split]:
                b=data[index]; digest=hashlib.sha256()
                for key in ['gene_ids','value_bins','splice_flags']: digest.update(b[key].numpy().tobytes())
                v=b['value_bins'];paired=(v[:,::2]>0)&(v[:,1::2]>0)
                manifest.append(dict(cell_index=index,cell_id=b['cell_id'],split=split,genes=b['gene_ids'].numel()//2,
                                     paired_genes=int(paired.sum()),positive_tokens=int((v>0).sum()),
                                     single_modality_tokens=int((v>0).sum())-2*int(paired.sum()),
                                     input_sha256=digest.hexdigest()))
        frame=pd.DataFrame(manifest).sort_values('cell_index')
        signature_cfg=json.loads(json.dumps(cfg)); signature_cfg.pop('_config_path',None)
        signature_cfg.get('align',{}).pop('resume',None)
        identity=dict(config=signature_cfg, checkpoint_sha256=sha256(cfg['checkpoint']),
                      vocab_sha256=sha256(cfg['vocab']), manifest_sha256=hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest())
        if formal:
            identity['implementation_sha256']={name:sha256(Path(__file__).parent/name)
                for name in ['joint_training.py','joint_reconstruction.py','joint_qc.py','joint_batch.py']}
        if cfg['data'].get('token_path'):
            identity['token_source_sha256']=sha256(Path(cfg['data']['token_path']))
        if (out/'run_config.json').exists():
            if json.loads((out/'run_config.json').read_text()) != identity: raise ValueError('Output identity mismatch; use a new tag')
        frame.to_csv(out/'cell_manifest.csv',index=False)
        pd.DataFrame({'source_gene':data.names,'vocab_id':data.mapping}).to_csv(out/'gene_manifest.csv',index=False)
        write_json(out/'run_config.json',identity); write_json(out/'split.json',fold)
        if device.type=='cuda':
            from .ablations import resource_preflight
            # Reject ambiguous logical device remapping, and permanently exclude GPU 1.
            if os.environ.get('CUDA_VISIBLE_DEVICES') or device.index not in (0,2,3):
                raise ValueError('Use physical cuda:0/2/3 with CUDA_VISIBLE_DEVICES unset')
            props=torch.cuda.get_device_properties(device)
            if 'A100' not in props.name or props.total_memory<75*2**30: raise ValueError('80GB A100 required')
            write_json(out/'status.json',dict(status='gpu_preflight',pid=os.getpid(),checkpoint_eligible=False))
            attempt = 0
            while True:
                passed,trace,reason=resource_preflight(device.index,31,10,minimum_free_mb=61440,maximum_mean_utilization=10)
                trace.to_csv(out/'preflight.csv',index=False)
                if passed: break
                if not ac.get('resume_extension'): raise RuntimeError(reason)
                attempt += 1
                trace.to_csv(out/f'preflight_wait_{attempt:03d}.csv',index=False)
                write_json(out/'status.json',dict(status='waiting_for_gpu',pid=os.getpid(),device=str(device),
                                                attempt=attempt,reason=reason))
                time.sleep(30)
            torch.cuda.set_per_process_memory_fraction(60*2**30/props.total_memory,device)
        torch.set_num_threads(4); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if ac.get('full_length_smoke') and not ac.get('resume'):
            write_json(out/'status.json',dict(status='full_length_training_smoke',pid=os.getpid()))
            # Resource feasibility only: no labels, discard all updates and RNG.
            # Include the longest source cell, even if held out from fitting.
            lengths={i:data[i]['value_bins'].numel() for i in range(len(data.cells))}
            ordered=sorted(lengths,key=lengths.get)
            smoke_model=JointModel.from_checkpoint(cfg['checkpoint']).to(device).train()
            smoke_opt=torch.optim.AdamW([{'params':smoke_model.encoder.parameters(),'lr':1e-5},
                                         {'params':smoke_model.adapter.parameters(),'lr':1e-4}],weight_decay=1e-4)
            smoke_disc=torch.optim.AdamW(smoke_model.discriminator.parameters(),lr=1e-4,weight_decay=1e-4)
            smoke=[]
            for i in [ordered[len(ordered)//2],ordered[-1]]:
                b=to_device(data[i],device);smoke_opt.zero_grad(set_to_none=True)
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                start=time.monotonic();m=backward_cell(smoke_model,b,smoke_disc,.02,.3,divisor=1,seed=seed)
                torch.nn.utils.clip_grad_norm_(smoke_model.main_parameters(),1,error_if_nonfinite=True);smoke_opt.step()
                if device.type=='cuda':torch.cuda.synchronize(device)
                smoke.append(dict(cell_id=b['cell_id'],tokens=lengths[i],seconds=time.monotonic()-start,
                    peak_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,**m))
            pd.DataFrame(smoke).to_csv(out/'full_length_smoke.csv',index=False)
            del smoke_model,smoke_opt,smoke_disc,b
            if device.type=='cuda':torch.cuda.empty_cache()
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=JointModel.from_checkpoint(cfg['checkpoint']).to(device)
        model.activation_checkpointing=ac.get('activation_checkpointing',True)
        model.evaluation_batch_size=int(ac.get('eval_batch_size',microbatch))
        optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':1e-5},
                                    {'params':model.adapter.parameters(),'lr':1e-4}],weight_decay=1e-4)
        disc=torch.optim.AdamW(model.discriminator.parameters(),lr=1e-4,weight_decay=1e-4)
        first=0; resume=ac.get('resume')
        if resume:
            first=restore_state(Path(resume),model,optimizer,disc,identity,ac.get('resume_execution_migration',False),
                                ac.get('resume_extension',False))
            if ac.get('resume_execution_migration') or ac.get('resume_extension'):
                write_json(out/'migration.json',dict(source_checkpoint=str(resume),source_sha256=sha256(resume),
                    resumed_epoch=first,checked_inputs=True,checked_scientific_config=True,
                    extension=bool(ac.get('resume_extension')),cuda_rng_transfer='source active device to target active device',
                    current_identity=identity,old_identity=torch.load(resume,map_location='cpu',weights_only=False)['identity']))
        elif list(out.glob('epoch_*.pt')): raise ValueError('Existing checkpoints: explicit resume required')
        else:
            write_json(out/'status.json',dict(status='raw_validation',pid=os.getpid(),checkpoint_eligible=False))
            baseline=evaluate(model,data,fold['validation'],device,seed,out,'raw')
            write_json(out/'raw_validation.json',baseline)
        if formal:
            from .joint_qc import collect,compare,selection_key
            if not resume:
                write_json(out/'status.json',dict(status='raw_qc',pid=os.getpid(),formal_training_started=False))
                raw_qc,raw_state,raw_cells=collect(model,data,fold['validation'],device,seed,out,'raw')
            else:
                raw_qc=json.loads((out/'raw_qc.json').read_text())
                raw_state=np.load(out/'raw_qc_state.npy'); raw_cells=pd.read_csv(out/'raw_qc_cells.csv')
            baseline=json.loads((out/'raw_validation.json').read_text())
            selection=json.loads((out/'selection.json').read_text()) if (out/'selection.json').exists() else dict(
                selected_epoch=None,selected_checkpoint=None,selection_key=None,protocol='internal_validation_only',
                thresholds=dict(coverage=.999,reconstruction_ratio=1.05,rank_ratio=.9,iqr_ratio=.5,knn_overlap=.9,separation_ratio=.9))
            if resume:
                # A crash after atomic checkpoint save may precede the selection write.
                # Re-audit that completed epoch before proceeding; never skip its QC.
                previous=first-1
                qc,qs,qcells=collect(model,data,fold['validation'],device,seed,out,f'epoch_{previous:03d}')
                val=json.loads((out/f'epoch_{previous:03d}_validation.json').read_text())
                comparison=compare(raw_qc,qc,raw_state,qs,raw_cells,qcells,baseline,val)
                key=selection_key(qc,comparison,val,previous)
                comparison['checkpoint_eligible']=key is not None
                write_json(out/f'epoch_{previous:03d}_eligibility.json',comparison)
                if key is not None and (selection['selection_key'] is None or key<tuple(selection['selection_key'])):
                    selection.update(selected_epoch=previous,selected_checkpoint=f'epoch_{previous:03d}.pt',selection_key=list(key))
                write_json(out/'selection.json',selection)
                for completed in range(first):
                    update_loss_selection(out, ac, completed)
        accumulate=32; resources=[]
        for epoch in range(first,count):
            model.train(); start=time.monotonic(); metrics=[]
            # Length buckets change order, never membership; all cells exactly once.
            order=np.array(fold['train']); rng=np.random.default_rng(seed+epoch); rng.shuffle(order)
            buckets=[order[i:i+64] for i in range(0,len(order),64)]
            lengths=frame.set_index('cell_index').genes.to_dict()
            order=np.concatenate([sorted(b,key=lambda i:lengths[i]) for b in buckets])
            for offset in range(0,len(order),accumulate):
                chunk=order[offset:offset+accumulate]; optimizer.zero_grad(set_to_none=True)
                for mbstart in range(0,len(chunk),microbatch):
                    subset=chunk[mbstart:mbstart+microbatch]
                    cells=[data[int(i)] for i in subset]
                    if ac.get('batch_backend','serial')=='serial':
                        if microbatch!=1:raise ValueError('Serial backend requires batch 1')
                        b=to_device(cells[0],device)
                        records=[backward_cell(model,b,disc,float(ac.get('adv_weight',.1))*min((epoch+1)/5,1),
                                              float(ac.get('rec_weight',.3)),len(chunk),seed,epoch)]
                    else:
                        from .joint_batch import collate,move,backward_batch
                        b=move(collate(cells),device)
                        records=backward_batch(model,b,disc,float(ac.get('adv_weight',.1))*min((epoch+1)/5,1),
                                               float(ac.get('rec_weight',.3)),len(chunk),seed,epoch)
                    for cell,row in zip(cells,records):
                        if not all(np.isfinite(list(row.values()))):raise FloatingPointError('Nonfinite training metrics')
                        metrics.append(dict(epoch=epoch,cell_id=cell['cell_id'],**row))
                grad=torch.nn.utils.clip_grad_norm_(model.main_parameters(),1.0,error_if_nonfinite=True)
                optimizer.step()
                resource=dict(epoch=epoch,cells_completed=offset+len(chunk),seconds=time.monotonic()-start,grad_norm=float(grad))
                if device.type=='cuda':
                    resource.update(allocated_gib=torch.cuda.memory_allocated(device)/2**30,
                                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                                    reserved_gib=torch.cuda.memory_reserved(device)/2**30)
                resources.append(resource)
                pd.DataFrame(resources).to_csv(out/'resources.csv',index=False)
                write_json(out/'status.json',dict(status='training_formal' if formal else 'training_pilot',
                           pid=os.getpid(),checkpoint_eligible=False,formal_training_started=formal,**resource))
            pd.DataFrame(metrics).to_csv(out/f'epoch_{epoch:03d}_train.csv',index=False)
            val=evaluate(model,data,fold['validation'],device,seed,out,f'epoch_{epoch:03d}')
            write_json(out/f'epoch_{epoch:03d}_validation.json',val)
            save_state(out/f'epoch_{epoch:03d}.pt',model,optimizer,disc,epoch,identity)
            if formal:
                write_json(out/'status.json',dict(status='epoch_qc',epoch=epoch,pid=os.getpid(),formal_training_started=True))
                qc,qs,qcells=collect(model,data,fold['validation'],device,seed,out,f'epoch_{epoch:03d}')
                comparison=compare(raw_qc,qc,raw_state,qs,raw_cells,qcells,baseline,val)
                key=selection_key(qc,comparison,val,epoch)
                comparison['checkpoint_eligible']=key is not None
                write_json(out/f'epoch_{epoch:03d}_eligibility.json',comparison)
                if key is not None and (selection['selection_key'] is None or key<tuple(selection['selection_key'])):
                    selection.update(selected_epoch=epoch,selected_checkpoint=f'epoch_{epoch:03d}.pt',selection_key=list(key))
                write_json(out/'selection.json',selection)
                update_loss_selection(out, ac, epoch)
        if formal:
            if ac.get('selection_policy')=='weighted_train_postwarmup' and (out/'loss_selection.json').exists():
                selection=json.loads((out/'loss_selection.json').read_text())
            write_json(out/'status.json',dict(status='formal_fit_complete' if selection['selected_epoch'] is not None
                       else 'formal_fit_no_eligible_checkpoint',pid=os.getpid(),epochs=count,
                       formal_training_started=True,outer_test_evaluated=False,selection=selection))
            return out
        write_json(out/'status.json',dict(status='pilot_complete',pid=os.getpid(),epochs=count,checkpoint_eligible=False,
                    formal_training_started=False,outer_test_evaluated=False,
                    pending=['independent modality probe','matched-null gates','complete anti-collapse gates','formal selection']))
        return out
    except Exception as exc:
        write_json(out/'status.json',dict(status='failed',error=str(exc),pid=os.getpid(),checkpoint_eligible=False,
                    recovery='Resume only from the last complete epoch checkpoint; partial epoch must be replayed'))
        raise
    finally:
        lock.unlink(missing_ok=True)

"""Full-token batching with one discriminator update per original cell."""
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from torch.func import functional_call
from .joint_reconstruction import mask_tokens, reconstruction, discriminator_loss


def collate(cells):
    if not cells: raise ValueError('Empty batch')
    lengths=[b['value_bins'].numel() for b in cells]
    width=max(lengths)
    result={k:torch.zeros((len(cells),width),dtype=torch.long) for k in ['gene_ids','value_bins','splice_flags']}
    for row,(b,n) in enumerate(zip(cells,lengths)):
        for k in result: result[k][row,:n]=b[k].reshape(-1)
    result.update(cell_ids=[str(b['cell_id']) for b in cells],lengths=lengths,
                  valid_mask=result['value_bins'].gt(0))
    return result


def batch_masks(batch,ratio,seed,epoch,replica=0):
    v=batch['value_bins']; result=torch.zeros_like(v,dtype=torch.bool)
    for row,(cell,n) in enumerate(zip(batch['cell_ids'],batch['lengths'])):
        result[row,:n]=mask_tokens(v[row:row+1,:n],ratio,seed,epoch,cell,replica)[0]
    return result


def move(batch,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def discriminator_steps(model,z,v,s,optimizer):
    """Snapshot each post-update discriminator, never share mutable storage."""
    snapshots=[];losses=[]
    model.discriminator.requires_grad_(True)
    for row in range(len(v)):
        optimizer.zero_grad(set_to_none=True)
        loss=discriminator_loss(model,z[row:row+1].detach(),v[row:row+1],s[row:row+1])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite discriminator loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(),1,error_if_nonfinite=True)
        optimizer.step()
        snapshots.append({k:x.detach().clone() for k,x in model.discriminator.state_dict().items()})
        losses.append(loss.detach())
    optimizer.zero_grad(set_to_none=True)
    model.discriminator.requires_grad_(False)
    return snapshots,losses


def snapshot_confusion(model,z,v,snapshots):
    result=[]
    for row,params in enumerate(snapshots):
        x=F.normalize(z[row,v[row]>0],dim=-1)
        logits=functional_call(model.discriminator,params,(x,))
        logp=F.log_softmax(logits,dim=1)
        result.append(-.5*(logp[:,0]+logp[:,1]).mean())
    return result


def backward_batch(model,batch,disc_optimizer,adv_weight,rec_weight,group_cells,seed=42,epoch=0):
    g,v,s=[batch[k] for k in ['gene_ids','value_bins','splice_flags']]
    if group_cells<len(v):raise ValueError('Microbatch exceeds update group')
    amp=lambda:torch.autocast(g.device.type,dtype=torch.bfloat16,enabled=g.is_cuda)
    metrics=[{} for _ in range(len(v))]
    try:
        if adv_weight:
            with torch.no_grad(),amp():_,detached=model.hidden(g,v,s)
            with amp():snapshots,disc_losses=discriminator_steps(model,detached,v,s,disc_optimizer)
            del detached
            with amp():
                _,z=model.hidden(g,v,s)
                adv=snapshot_confusion(model,z,v,snapshots)
                loss=torch.stack(adv).sum()*adv_weight/group_cells
            loss.backward()
            for row in range(len(v)):
                metrics[row].update(loss_adv=adv[row].detach(),loss_disc=disc_losses[row])
            del z,loss,adv,snapshots,disc_losses
        if rec_weight:
            mask=batch_masks(batch,model.encoder.mask_ratio,seed,epoch)
            masked=v.clone();masked[mask]=model.encoder.mask_token
            with amp():
                _,z=model.hidden(g,masked,s)
                logits=model.encoder.head_gene(z)
                records=[reconstruction(logits[r:r+1],v[r:r+1],s[r:r+1],mask[r:r+1]) for r in range(len(v))]
                loss=torch.stack([r['loss_rec'] for r in records]).sum()*rec_weight/group_cells
            loss.backward()
            for r,record in enumerate(records):metrics[r].update({k:x.detach() for k,x in record.items()})
        # One synchronization for the metric block, not per token/cell loss.
        keys=list(metrics[0]);values=torch.stack([torch.stack([r[k] for k in keys]) for r in metrics]).float().cpu().numpy()
        return [dict(zip(keys,map(float,row))) for row in values]
    finally:model.discriminator.requires_grad_(True)


@torch.no_grad()
def encoded_cells(model,data,indices,device,batch_size=1):
    """Yield original cell-shaped views of a batched complete-input encoding."""
    for offset in range(0,len(indices),batch_size):
        cells=[data[i] for i in indices[offset:offset+batch_size]]
        b=move(collate(cells),device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            h,z=model.hidden(b['gene_ids'],b['value_bins'],b['splice_flags'])
        for row,(cell,n) in enumerate(zip(cells,b['lengths'])):
            yield move(cell,device),h[row:row+1,:n],z[row:row+1,:n]


@torch.no_grad()
def evaluate_batched(model,data,indices,device,seed,out,label,batch_size):
    import numpy as np
    import pandas as pd
    from .runtime import cosine_distance
    model.eval();rows=[];states=[]
    for offset in range(0,len(indices),batch_size):
        b=move(collate([data[i] for i in indices[offset:offset+batch_size]]),device)
        g,v,s=[b[k] for k in ['gene_ids','value_bins','splice_flags']]
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            _,z=model.hidden(g,v,s)
        batchrows=[]
        for r,n in enumerate(b['lengths']):
            paired=(v[r,:n:2]>0)&(v[r,1:n:2]>0)
            if not paired.any():raise ValueError('No valid paired genes')
            u=z[r,:n:2][paired].float();sp=z[r,1:n:2][paired].float()
            d=cosine_distance(u,sp)
            if not torch.isfinite(d).all() or d.min()<-1e-6 or d.max()>2+1e-6:raise ValueError('Invalid distances')
            states.append(((u+sp)/2).mean(0).cpu().numpy())
            batchrows.append(dict(cell_id=b['cell_ids'][r],genes=n//2,paired_genes=int(paired.sum()),
                distance_mean=float(d.mean()),distance_median=float(d.median()),
                distance_iqr=float(torch.quantile(d,.75)-torch.quantile(d,.25))))
        del z
        for replica in range(2):
            mask=batch_masks(b,model.encoder.mask_ratio,seed,0,replica)
            masked=v.clone();masked[mask]=model.encoder.mask_token
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                _,z=model.hidden(g,masked,s);logits=model.encoder.head_gene(z)
                for r in range(len(v)):
                    rec=reconstruction(logits[r:r+1],v[r:r+1],s[r:r+1],mask[r:r+1])
                    for k,x in rec.items():batchrows[r][k]=batchrows[r].get(k,0)+float(x)/2
            del z,logits
        rows.extend(batchrows)
    frame=pd.DataFrame(rows)
    if not np.isfinite(frame.select_dtypes('number').to_numpy()).all():raise FloatingPointError('Nonfinite validation')
    frame.to_csv(out/f'{label}_validation_cells.csv',index=False)
    np.save(out/f'{label}_validation_state.npy',np.stack(states))
    return {k:float(frame[k].mean()) for k in frame.select_dtypes('number').columns}

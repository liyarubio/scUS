"""All-gene joint reconstruction primitives; no cached embedding training."""
from __future__ import annotations
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .align import ModalityDiscriminator, _confusion
from .models import MaskedModel

class IdentityResidualAdapter(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,dim))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
        self.alpha_logit=nn.Parameter(torch.tensor(-2.944438979))

    def forward(self,h):
        return h+torch.sigmoid(self.alpha_logit)*self.network(h)

class JointModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder=encoder
        self.encoder.requires_grad_(True)
        self.adapter=IdentityResidualAdapter(encoder.head_gene.in_features)
        self.discriminator=ModalityDiscriminator(encoder.head_gene.in_features)
        self.activation_checkpointing=True

    @classmethod
    def from_checkpoint(cls,path):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if 'mask_ratio' not in payload.get('hyper_parameters',{}):
            raise ValueError('Checkpoint lacks historical mask_ratio; audit original config first')
        encoder=MaskedModel(**payload['hyper_parameters'])
        encoder.load_state_dict(payload['state_dict'],strict=True)
        return cls(encoder)

    def hidden(self,g,v,s):
        h=self.encoder.embed(g,v,s)
        from torch.nn.attention import sdpa_kernel, SDPBackend
        def execute(layer,h,padding):
            # Refuse quadratic math-attention fallback on long CUDA sequences.
            backend=sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION]) if h.is_cuda else nullcontext()
            with backend:
                return layer(h,attn_mask=padding)
        for layer in self.encoder.encoder.layers:
            if self.training and self.activation_checkpointing:
                h=checkpoint(execute,layer,h,v.eq(0),use_reentrant=False)
            else:
                # Historical custom FlashAttention path passes dropout_p directly.
                # Honor eval mode without changing the historical model globally.
                dropout=layer.attn.dropout_p
                if not self.training: layer.attn.dropout_p=0.0
                try: h=execute(layer,h,v.eq(0))
                finally: layer.attn.dropout_p=dropout
        return h,self.adapter(h)

    def main_parameters(self):
        return list(self.encoder.parameters())+list(self.adapter.parameters())

def mask_tokens(values,ratio,seed,epoch,cell_id,replica=0):
    key=f'{seed}:{epoch}:{cell_id}:{replica}'.encode()
    generator=torch.Generator(device='cpu').manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8],'little')%(2**63-1))
    eligible=values.detach().cpu().gt(0)
    mask=(torch.rand(values.shape,generator=generator)<ratio)&eligible
    for row in range(len(mask)):
        indices=torch.where(eligible[row])[0]
        if len(indices) and not mask[row].any():
            mask[row,indices[torch.randint(len(indices),(1,),generator=generator)]]=True
    return mask.to(values.device)

def reconstruction(logits,targets,flags,mask):
    """Equal modality means within each cell, then equal cell means."""
    values={k:[] for k in ['loss_u','loss_s','loss_rec','loss_token_weighted','accuracy_u','accuracy_s']}
    for row in range(len(targets)):
        losses=[]; counts=[]
        for modality,name in [(0,'u'),(1,'s')]:
            use=mask[row]&targets[row].gt(0)&flags[row].eq(modality)
            n=int(use.sum()); counts.append(n)
            loss=F.cross_entropy(logits[row,use].float(),targets[row,use]) if n else logits[row].sum()*0
            losses.append(loss); values['loss_'+name].append(loss)
            values['accuracy_'+name].append((logits[row,use].argmax(-1)==targets[row,use]).float().mean() if n else loss.detach())
        values['loss_rec'].append((losses[0]+losses[1])/2)
        values['loss_token_weighted'].append(sum(x*n for x,n in zip(losses,counts))/max(sum(counts),1))
    return {key:torch.stack(value).mean() for key,value in values.items()}

def discriminator_loss(model,z,values,flags):
    result=[]
    for row in range(len(values)):
        for modality in [0,1]:
            use=values[row].gt(0)&flags[row].eq(modality)
            if use.any():
                prediction=model.discriminator(F.normalize(z[row,use],dim=-1))
                result.append(F.cross_entropy(prediction,torch.full((int(use.sum()),),modality,dtype=torch.long,device=z.device)))
    if not result: raise ValueError('No valid modality tokens')
    return torch.stack(result).mean()

def backward_cell(model,batch,disc_optimizer,adv_weight,rec_weight,divisor=32,seed=42,epoch=0):
    """Sequential graphs; caller accumulates and steps main optimizer."""
    g,v,s=batch['gene_ids'],batch['value_bins'],batch['splice_flags']
    if len(g)!=1: raise ValueError('Full-gene training requires microbatch 1')
    device=g.device
    amp=lambda:torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda')
    metrics={}
    if adv_weight:
        model.discriminator.requires_grad_(True)
        with torch.no_grad(),amp(): _,z=model.hidden(g,v,s)
        disc_optimizer.zero_grad(set_to_none=True)
        with amp(): dl=discriminator_loss(model,z.detach(),v,s)
        if not torch.isfinite(dl): raise FloatingPointError('Nonfinite discriminator loss')
        dl.backward()
        torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(),1.0,error_if_nonfinite=True)
        disc_optimizer.step(); disc_optimizer.zero_grad(set_to_none=True)
        del z
        model.discriminator.requires_grad_(False)
        with amp():
            _,z=model.hidden(g,v,s)
            adv=_confusion(model.discriminator,F.normalize(z[v.gt(0)],dim=-1))
        (adv*adv_weight/divisor).backward()
        metrics.update(loss_adv=float(adv.detach()),loss_disc=float(dl.detach()))
        del z,adv,dl
    if rec_weight:
        mask=mask_tokens(v,model.encoder.mask_ratio,seed,epoch,batch['cell_id'])
        masked=v.clone(); masked[mask]=model.encoder.mask_token
        with amp():
            _,z=model.hidden(g,masked,s)
            losses=reconstruction(model.encoder.head_gene(z),v,s,mask)
        (rec_weight*losses['loss_rec']/divisor).backward()
        metrics.update({k:float(x.detach()) for k,x in losses.items()})
    model.discriminator.requires_grad_(True)
    return metrics

class AllGeneCache:
    """Input-bin cache only; U/S-union eligibility, no top-k truncation."""
    def __init__(self,path,vocab):
        path=Path(path); self.path=path
        lookup={k.upper():v for k,v in json.loads(Path(vocab).read_text()).items()}
        self.names=np.load(path/'gene_names.npy',allow_pickle=True).astype(str)
        self.mapping=np.array([lookup.get(g.upper(),0) for g in self.names])
        self.cells=np.load(path/'barcodes.npy',allow_pickle=True).astype(str)
        ci=np.load(path/'cell_idx.int32.npy',mmap_mode='r')
        if len(set(self.cells)) != len(self.cells):
            raise ValueError('Duplicate cache cell IDs')
        if len(ci) == 0 or ci.min() < 0 or ci.max() >= len(self.cells) or np.any(ci[1:] < ci[:-1]):
            raise ValueError('Cache pairs must be grouped by valid ascending cell index')
        self.ptr=np.r_[0,np.cumsum(np.bincount(ci,minlength=len(self.cells)))]
        self.g=np.load(path/'gene_idx.int32.npy',mmap_mode='r')
        self.u=np.load(path/'u_bin.uint8.npy',mmap_mode='r'); self.s=np.load(path/'s_bin.uint8.npy',mmap_mode='r')
        if not (len(ci)==len(self.g)==len(self.u)==len(self.s)):
            raise ValueError('Cache pair arrays have different lengths')
        if self.g.min()<0 or self.g.max()>=len(self.mapping):
            raise ValueError('Invalid cache gene index')
        if self.u.max()>15 or self.s.max()>15:
            raise ValueError('Input bins must be in 0..15 (no premasked targets)')

    def __getitem__(self,index):
        from .ablations import interleave_tokens
        sl=slice(self.ptr[index],self.ptr[index+1]); g=self.mapping[self.g[sl]]
        u=np.asarray(self.u[sl]); s=np.asarray(self.s[sl]); valid=(g>0)&((u>0)|(s>0))
        order=np.argsort(g[valid]); g=g[valid][order]; u=u[valid][order]; s=s[valid][order]
        if len(np.unique(g))!=len(g): raise ValueError('Vocabulary collision requires explicit resolution')
        if not len(g): raise ValueError('Cell has no eligible genes')
        tg,tv,ts=interleave_tokens(g[None],u[None],s[None])
        return dict(gene_ids=torch.from_numpy(tg),value_bins=torch.from_numpy(tv),splice_flags=torch.from_numpy(ts),cell_id=self.cells[index])

def biological_folds(cell_ids,seed=42):
    """Outer sample holdout; hashed cell split exclusively inside training sample."""
    ids=list(map(str,cell_ids))
    if len(set(ids))!=len(ids): raise ValueError('Duplicate cell IDs')
    groups=[c.split(':',1)[0] for c in ids]
    if len(set(groups))!=2: raise ValueError('Forebrain protocol requires exactly two sample groups')
    folds=[]
    for heldout in sorted(set(groups)):
        train=[i for i,g in enumerate(groups) if g!=heldout]
        train.sort(key=lambda i:hashlib.sha256(f'{seed}:{ids[i]}'.encode()).hexdigest())
        n=max(1,int(np.ceil(.2*len(train))))
        if n==len(train): raise ValueError('Insufficient cells for internal validation')
        folds.append(dict(heldout_sample=heldout,train=train[n:],validation=train[:n],test=[i for i,g in enumerate(groups) if g==heldout]))
    return folds


class FullTokenDataset:
    """Historical interleaved token source, including single-modality context.

    Zero-value partners remain padding keys; positive partners are never dropped.
    No binning, rank recomputation, gene cap, or pair-cache reconstruction.
    """
    def __init__(self,path,vocab):
        self.path=Path(path)
        self.shard=torch.load(self.path,map_location='cpu',weights_only=False)
        lookup=json.loads(Path(vocab).read_text())
        self.cells=np.asarray(self.shard['obs_barcode'],dtype=str)
        if len(set(self.cells))!=len(self.cells):raise ValueError('Duplicate token cell IDs')
        observed=set()
        for i in range(len(self.cells)):
            b=self[i];observed.update(b['gene_ids'][0,::2].tolist())
        inverse={int(v):k for k,v in lookup.items()}
        if any(g<=0 or g not in inverse for g in observed):raise ValueError('Unknown gene ID')
        self.mapping=np.array(sorted(observed),dtype=np.int64)
        self.names=np.array([inverse[g] for g in self.mapping])

    def __getitem__(self,i):
        arrays=[torch.as_tensor(self.shard[k][i]).long().reshape(1,-1)
                for k in ['gene_ids','value_bins','splice_flags']]
        g,v,s=arrays
        if g.shape!=v.shape or g.shape!=s.shape or g.numel()%2:raise ValueError('Malformed interleaved tokens')
        if not torch.equal(g[:,::2],g[:,1::2]) or not (s[:,::2]==0).all() or not (s[:,1::2]==1).all():
            raise ValueError('Historical U/S interleaving mismatch')
        if (v<0).any() or (v>15).any() or (g<=0).any():raise ValueError('Invalid token values')
        if not ((v[:,::2]>0)|(v[:,1::2]>0)).all():raise ValueError('Gene with no valid modality')
        if g[:,::2].unique().numel()!=g.numel()//2:raise ValueError('Duplicate gene pair')
        return dict(gene_ids=g,value_bins=v,splice_flags=s,cell_id=self.cells[i])


def load_joint_data(cfg):
    settings=cfg['data']
    if settings.get('token_path'):
        if settings.get('cache_path'):raise ValueError('Specify token_path or cache_path, not both')
        return FullTokenDataset(settings['token_path'],cfg['vocab'])
    return AllGeneCache(settings['cache_path'],cfg['vocab'])

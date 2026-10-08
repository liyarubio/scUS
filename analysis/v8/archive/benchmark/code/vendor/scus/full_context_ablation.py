"""Protected-target interventions using full historical token context."""
import hashlib
import numpy as np
import torch

CONDITIONS=['full','matched_context','context_shuffle','target_panel_only','pair_only']

def split_target(cell,targets):
    g=cell['gene_ids'][0,::2].cpu().numpy()
    v=cell['value_bins'][0].cpu().numpy().reshape(-1,2)
    lookup={int(x):i for i,x in enumerate(g)}
    pos=np.array([lookup[int(x)] for x in targets]);rest=np.flatnonzero(~np.isin(g,targets))
    if not (v[pos]>0).all():raise ValueError('All protected targets require both modalities')
    return g,v,pos,rest

def make_cell(g,v,cell_id):
    return dict(gene_ids=torch.as_tensor(np.repeat(g,2)[None],dtype=torch.long),
        value_bins=torch.as_tensor(np.asarray(v).reshape(1,-1),dtype=torch.long),
        splice_flags=torch.as_tensor(np.tile([0,1],len(g))[None],dtype=torch.long),cell_id=cell_id)

def construct(cell,targets,condition,donor=None,seed=42):
    if condition not in CONDITIONS:raise ValueError(condition)
    g,v,pos,rest=split_target(cell,targets);tg=g[pos];tv=v[pos]
    if condition=='pair_only':return [make_cell(tg[i:i+1],tv[i:i+1],cell['cell_id']) for i in range(len(tg))]
    if condition=='target_panel_only':return make_cell(tg,tv,cell['cell_id'])
    if condition=='matched_context':
        if donor is None or donor['cell_id']==cell['cell_id']:raise ValueError('Nonself donor required')
        dg=donor['gene_ids'][0,::2].cpu().numpy();dv=donor['value_bins'][0].cpu().numpy().reshape(-1,2)
        keep=~np.isin(dg,tg);bg,bv=dg[keep],dv[keep]
    else:
        bg,bv=g[rest],v[rest].copy()
        if condition=='context_shuffle':
            key=int.from_bytes(hashlib.sha256(f'{seed}:{cell["cell_id"]}:context'.encode()).digest()[:8],'little')
            bv=bv[np.random.default_rng(key).permutation(len(rest))]
    # Common target prefix; no position embedding in scUS. All background genes
    # remain in original source order except the deliberately shuffled values.
    result=make_cell(np.r_[tg,bg],np.concatenate([tv,bv]),cell['cell_id'])
    if not np.array_equal(result['value_bins'][0,:2*len(tg)].numpy().reshape(-1,2),tv):raise AssertionError('Target values changed')
    if result['gene_ids'][0,::2].unique().numel()!=len(tg)+len(bg):raise AssertionError('Duplicate gene')
    return result

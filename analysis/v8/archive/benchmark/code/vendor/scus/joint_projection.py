"""Re-encode full input with the QC-selected jointly updated model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .artifacts import sha256
from .runtime import stage_directory,write_json,cosine_distance
from .joint_reconstruction import JointModel,load_joint_data
from .joint_training import to_device


@torch.no_grad()
def project(cfg,tag=None,start=0,stop=None,device_name=None,fit_tag=None,prepare_tag=None):
    source=stage_directory(cfg,'align-fit',fit_tag or tag)
    loss_primary=cfg['align'].get('selection_policy')=='weighted_train_postwarmup'
    selection_file=source/('loss_selection.json' if loss_primary else 'selection.json')
    if not selection_file.exists():
        raise ValueError('Joint project requires a loss-selected checkpoint' if loss_primary
                         else 'Joint project requires a fully QC-eligible selected model')
    selection=json.loads(selection_file.read_text())
    epoch=selection.get('selected_epoch')
    if epoch is None: raise ValueError('No QC-eligible joint checkpoint')
    eligibility=json.loads((source/f'epoch_{epoch:03d}_eligibility.json').read_text())
    if not loss_primary and not eligibility['checkpoint_eligible']: raise ValueError('Selected checkpoint failed QC')
    ck=source/f'epoch_{epoch:03d}.pt'
    payload=torch.load(ck,map_location='cpu',weights_only=False)
    if payload['identity']['checkpoint_sha256']!=sha256(cfg['checkpoint']): raise ValueError('Initialization hash mismatch')
    if payload['identity']['vocab_sha256']!=sha256(cfg['vocab']): raise ValueError('Vocabulary hash mismatch')
    data=load_joint_data(cfg); stop=len(data.cells) if stop is None else stop
    if not 0<=start<stop<=len(data.cells): raise ValueError('Invalid cell range')
    device=torch.device(device_name or cfg.get('device','cpu'))
    if device.type=='cuda':
        import os
        from .ablations import resource_preflight
        if os.environ.get('CUDA_VISIBLE_DEVICES') or device.index not in (0,2,3): raise ValueError('Use physical GPU 0/2/3')
        ok,trace,reason=resource_preflight(device.index,31,10,minimum_free_mb=61440,maximum_mean_utilization=10)
        if not ok: raise RuntimeError(reason)
        torch.cuda.set_per_process_memory_fraction(60*2**30/torch.cuda.get_device_properties(device).total_memory,device)
    out=stage_directory(cfg,'align-project',tag)/f'cells_{start:06d}_{stop:06d}'
    out.mkdir(exist_ok=False)
    model=JointModel.from_checkpoint(cfg['checkpoint']).to(device);model.load_state_dict(payload['model']);model.eval()
    manifest=pd.read_csv(source/'cell_manifest.csv').set_index('cell_index')
    sizes=[]
    for i in range(start,stop):
        b=data[i]
        import hashlib
        digest=hashlib.sha256()
        for k in ['gene_ids','value_bins','splice_flags']: digest.update(b[k].numpy().tobytes())
        if digest.hexdigest()!=manifest.loc[i,'input_sha256']: raise ValueError('Projection input hash mismatch')
        v=b['value_bins'][0];sizes.append(int(((v[::2]>0)&(v[1::2]>0)).sum()))
    ptr=np.r_[0,np.cumsum(sizes)];np.save(out/'indptr.npy',ptr)
    genes=np.lib.format.open_memmap(out/'indices.npy',mode='w+',dtype='int32',shape=(int(ptr[-1]),))
    distances=np.lib.format.open_memmap(out/'data.npy',mode='w+',dtype='float32',shape=(int(ptr[-1]),))
    hs=[];zs=[]
    for row,i in enumerate(range(start,stop)):
        b=to_device(data[i],device);g,v,s=(b[k] for k in ['gene_ids','value_bins','splice_flags'])
        mask=(v[0,::2]>0)&(v[0,1::2]>0)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):h,z=model.hidden(g,v,s)
        d=cosine_distance(z[0,::2][mask],z[0,1::2][mask]).cpu().numpy()
        if not np.isfinite(d).all() or d.min()<-1e-6 or d.max()>2+1e-6: raise ValueError('Invalid projected distances')
        sl=slice(ptr[row],ptr[row+1]);genes[sl]=g[0,::2][mask].cpu().numpy();distances[sl]=d
        for value,store in [(h,hs),(z,zs)]:
            store.append(((value[0,::2][mask].float()+value[0,1::2][mask].float())/2).mean(0).cpu().numpy())
    genes.flush();distances.flush();np.save(out/'finetuned_h_state.npy',np.stack(hs));np.save(out/'aligned_z_state.npy',np.stack(zs))
    pd.DataFrame(dict(cell_index=range(start,stop),cell_id=data.cells[start:stop],paired_genes=sizes)).to_csv(out/'cells.csv',index=False)
    write_json(out/'status.json',dict(status='complete',checkpoint_sha256=sha256(ck),selected_epoch=epoch,
               csr_columns='original vocabulary IDs (not compact source gene positions)',encoder_frozen=False,
               input_scope='all historical source token genes' if cfg['data'].get('token_path') else 'all eligible source-cache genes',
               selection_policy=selection.get('policy','internal_validation_qc'),
               qc_passed=eligibility['checkpoint_eligible'],cells=stop-start,pairs=int(ptr[-1])))
    return out

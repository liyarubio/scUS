"""Label-free, full-input checkpoint audit on internal validation only.

Token sampling is for the independent probe, never for encoder input. Nulls
are different genes in the same cell matched by U/S bins and detection rate.
This null does not substitute for the additional cross-cell pairing control.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .align import _effective_rank, _knn_overlap
from .runtime import cosine_distance, write_json
from .joint_training import to_device


def matched_gene_donors(features):
    """Nearest nonself gene; standardized bin/detection covariates only."""
    x=np.asarray(features,float)
    if len(x)<2: raise ValueError('At least two genes required for a pairing null')
    x=(x-x.mean(0))/np.maximum(x.std(0),1e-8)
    d,nn=NearestNeighbors(n_neighbors=min(3,len(x))).fit(x).kneighbors(x)
    donors=np.array([next(j for j in row if j!=i) for i,row in enumerate(nn)])
    return donors,np.linalg.norm(x-x[donors],axis=1)


def independent_probe(u,s,groups,seed=42):
    """Disjoint validation-cell probe fit/test; no biological labels."""
    groups=np.asarray(groups); ids=np.unique(groups)
    ids=sorted(ids,key=lambda c:hashlib.sha256(f'{seed}:{c}'.encode()).hexdigest())
    if len(ids)<4: return {'probe_balanced_accuracy':None,'probe_auroc':None}
    train=np.isin(groups,ids[:len(ids)//2]); test=~train
    x=np.concatenate([u,s]); y=np.r_[np.zeros(len(u)),np.ones(len(s))]
    fit=np.tile(train,2); heldout=np.tile(test,2)
    probe=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=seed))
    probe.fit(x[fit],y[fit]); p=probe.predict_proba(x[heldout])[:,1]
    return dict(probe_balanced_accuracy=float(balanced_accuracy_score(y[heldout],p>=.5)),
                probe_auroc=float(roc_auc_score(y[heldout],p)),
                probe_train_cells=len(ids)//2,probe_test_cells=len(ids)-len(ids)//2)


@torch.no_grad()
def collect(model,data,indices,device,seed,out,label):
    model.eval(); detection={}
    for i in indices:
        b=data[i]; g=b['gene_ids'][0,::2].numpy(); v=b['value_bins'][0].numpy()
        for gene,u,s in zip(g,v[::2],v[1::2]):
            detection.setdefault(int(gene),np.zeros(2))[:] += [u>0,s>0]
    detection={g:v/len(indices) for g,v in detection.items()}
    rows=[]; states=[]; all_distance=[]; pu=[]; ps=[]; groups=[]; pair_cache=[]; cell_features=[]
    from .joint_batch import encoded_cells
    for b,_,z in encoded_cells(model,data,indices,device,getattr(model,'evaluation_batch_size',1)):
        g,v,s=(b[k] for k in ['gene_ids','value_bins','splice_flags'])
        paired=(v[0,::2]>0)&(v[0,1::2]>0)
        u=z[0,::2][paired].float(); sp=z[0,1::2][paired].float()
        genes=g[0,::2][paired].cpu().numpy()
        uv=v[0,::2][paired].cpu().numpy(); sv=v[0,1::2][paired].cpu().numpy()
        # CPU cache for QC only; encoder input remains the complete sequence.
        pair_cache.append((genes,torch.nn.functional.normalize(u,dim=-1).cpu().numpy(),
                           torch.nn.functional.normalize(sp,dim=-1).cpu().numpy()))
        cell_features.append([uv.mean(),sv.mean(),float((v[0,::2]>0).sum())/len(data.names),
                              float((v[0,1::2]>0).sum())/len(data.names)])
        donors,match_error=matched_gene_donors(np.column_stack([uv,sv,np.stack([detection[int(g)] for g in genes])]))
        true=cosine_distance(u,sp).cpu().numpy()
        null=cosine_distance(u,sp[torch.as_tensor(donors,device=device)]).cpu().numpy()
        state=((u+sp)/2).mean(0).cpu().numpy(); states.append(state)
        if not np.isfinite(np.r_[true,null,state]).all(): raise FloatingPointError('Nonfinite checkpoint output')
        if min(true.min(),null.min()) < -1e-6 or max(true.max(),null.max())>2+1e-6:
            raise ValueError('Cosine distance outside legal range')
        rows.append(dict(cell_id=b['cell_id'],paired_genes=len(genes),distance_mean=true.mean(),
                         null_mean=null.mean(),delta_mean=(null-true).mean(),
                         null_match_error_mean=match_error.mean(),state_norm=np.linalg.norm(state)))
        all_distance.append(true)
        key=int.from_bytes(hashlib.sha256(f'{seed}:{b["cell_id"]}'.encode()).digest()[:8],'little')
        chosen=np.random.default_rng(key).choice(len(u),min(64,len(u)),replace=False)
        pu.append(torch.nn.functional.normalize(u[chosen],dim=-1).cpu().numpy())
        ps.append(torch.nn.functional.normalize(sp[chosen],dim=-1).cpu().numpy()); groups.extend([b['cell_id']]*len(chosen))
    # Same-gene S from a different, expression/detection-matched cell.
    # Matching uses only input-bin summaries, never learned embeddings or labels.
    cell_donors,cell_error=matched_gene_donors(cell_features)
    for i,j in enumerate(cell_donors):
        gi,ui,si=pair_cache[i]; gj,_,sj=pair_cache[j]
        common,a,b=np.intersect1d(gi,gj,return_indices=True)
        if not len(common): raise ValueError('Matched cells have no shared paired genes')
        true=1-np.sum(ui[a]*si[a],axis=1); null=1-np.sum(ui[a]*sj[b],axis=1)
        rows[i].update(cell_null_delta=float((null-true).mean()),cell_null_pairs=len(common),
                       cell_null_donor=rows[j]['cell_id'],cell_null_match_error=float(cell_error[i]))
    frame=pd.DataFrame(rows); state=np.stack(states); distances=np.concatenate(all_distance)
    frame.to_csv(out/f'{label}_qc_cells.csv',index=False); np.save(out/f'{label}_qc_state.npy',state)
    delta=frame.delta_mean.to_numpy()
    result=dict(cells=len(frame),coverage=len(frame)/len(indices),
                pooled_nonzero_coverage=float(np.mean(np.linalg.norm(state,axis=1)>1e-8)),
                effective_rank=_effective_rank(state),distance_iqr=float(np.diff(np.quantile(distances,[.25,.75]))[0]),
                matched_shuffle_effect=float(delta.mean()/max(delta.std(ddof=1),1e-12)),
                matched_cell_shuffle_effect=float(frame.cell_null_delta.mean()/max(frame.cell_null_delta.std(ddof=1),1e-12)),
                matched_cell_pair_coverage=float(frame.cell_null_pairs.sum()/frame.paired_genes.sum()),
                null_definition='within-cell other gene; nearest standardized U/S bins and validation detection rates',
                null_match_error_mean=float(frame.null_match_error_mean.mean()))
    result.update(independent_probe(np.concatenate(pu),np.concatenate(ps),groups,seed))
    write_json(out/f'{label}_qc.json',result)
    return result,state,frame


def compare(raw,candidate,raw_state,state,raw_cells,cells,raw_rec,rec):
    if list(raw_cells.cell_id)!=list(cells.cell_id): raise ValueError('Cell alignment mismatch')
    if not np.array_equal(raw_cells.paired_genes,cells.paired_genes): raise ValueError('Pair coverage mismatch')
    overlap=_knn_overlap(raw_state,state)
    gates=dict(coverage=candidate['coverage']>=.999 and candidate['pooled_nonzero_coverage']>=.999,
               reconstruction_u=rec['loss_u']<=1.05*raw_rec['loss_u'],
               reconstruction_s=rec['loss_s']<=1.05*raw_rec['loss_s'],
               rank=candidate['effective_rank']>=.9*raw['effective_rank'],
               distance_iqr=candidate['distance_iqr']>=.5*raw['distance_iqr'],knn=overlap>=.9,
               matched_separation=raw['matched_shuffle_effect']>0 and candidate['matched_shuffle_effect']>0
               and candidate['matched_shuffle_effect']>=.9*raw['matched_shuffle_effect'])
    if 'matched_cell_shuffle_effect' in raw and 'matched_cell_shuffle_effect' in candidate:
        gates['matched_cell_separation']=bool(raw['matched_cell_shuffle_effect']>0 and
            candidate['matched_cell_shuffle_effect']>0 and
            candidate['matched_cell_shuffle_effect']>=.9*raw['matched_cell_shuffle_effect'])
    else:
        gates['matched_cell_separation']=False
    return dict(gates=gates,passed_measured_gates=all(gates.values()),knn_overlap=overlap,
                distance_mean_spearman=float(spearmanr(raw_cells.distance_mean,cells.distance_mean).statistic),
                checkpoint_eligible=False,
                note='Retrospective audit; eligibility is assigned only by the formal training protocol.')


def selection_key(qc,comparison,rec,epoch):
    """Preregistered order: eligible only, chance proximity, CE, earlier epoch."""
    probe=qc.get('probe_balanced_accuracy')
    if not comparison['passed_measured_gates'] or probe is None or not np.isfinite(probe): return None
    return (abs(float(probe)-.5),float(rec['loss_rec']),int(epoch))

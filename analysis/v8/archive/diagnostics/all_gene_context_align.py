"""Resumable all-eligible-gene context ablation, with smoke-first GPU gate."""
from pathlib import Path
import argparse, json, time, hashlib
import numpy as np
import pandas as pd
import torch
from scus.ablations import interleave_tokens, resource_preflight, query_gpu, _load_cache_panel
from scus.zero_shot import load_encoder
from scus.runtime import cosine_distance
from align_pairing_null import load_projector

parser=argparse.ArgumentParser()
parser.add_argument('--gpu',type=int,default=2)
parser.add_argument('--epochs',default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14')
parser.add_argument('--smoke-only',action='store_true')
args=parser.parse_args()
assert args.gpu in [0,2,3]
root=Path(__file__).resolve().parents[1]
base=root/'outputs/paper/pretraining_ablations'
out=base/'all_gene_context_align'; out.mkdir(exist_ok=True)
cfg=json.loads((base/'unified_context_align/run_config.json').read_text())
meta=json.loads((base/'forebrain_align_pairing/status.json').read_text())
cache=Path(meta['cache_dir']); vocab_path=Path('/data1/liyaru/proj_us/data/scBaseCount/gene_vocab.json')
vocab={k.upper():v for k,v in json.loads(vocab_path.read_text()).items()}
names=np.load(cache/'gene_names.npy',allow_pickle=True).astype(str)
mapping=np.array([vocab.get(g.upper(),0) for g in names])
ci=np.load(cache/'cell_idx.int32.npy',mmap_mode='r'); gi=np.load(cache/'gene_idx.int32.npy',mmap_mode='r')
ub=np.load(cache/'u_bin.uint8.npy',mmap_mode='r'); sb=np.load(cache/'s_bin.uint8.npy',mmap_mode='r')
ptr=np.r_[0,np.cumsum(np.bincount(ci,minlength=1720))]
manifest=pd.read_csv(base/'unified_context_align/cell_manifest.csv')
# Cross-type donor: pick, per manifest cell, a donor of a DIFFERENT cell type
# (cross-sample preferred) from the full 1720-cell cache, then map it to the
# manifest's local row so panel_inputs can index cells[...] consistently.
cell_types_all=np.load(cache/'cell_types.npy',allow_pickle=True).astype(str)
barcodes_all=np.load(cache/'barcodes.npy',allow_pickle=True).astype(str)
samples_all=np.array([b.split(':',1)[0] for b in barcodes_all])
def cross_type_donors(seed=42):
    n=len(manifest); rng=np.random.default_rng(seed); out=np.empty(n,np.int64)
    types=manifest.cell_type.astype(str).values; smp=samples_all[manifest.cell_index.values]
    for i in range(n):
        cand=np.flatnonzero(cell_types_all!=types[i])
        cross=cand[samples_all[cand]!=smp[i]]
        out[i]=cross[rng.integers(len(cross))] if len(cross) else cand[rng.integers(len(cand))]
    return out
_global_cross=cross_type_donors(42)
assert (cell_types_all[_global_cross]!=manifest.cell_type.astype(str).values).all()
assert (samples_all[_global_cross]!=samples_all[manifest.cell_index.values]).mean()>0.9
row_of=np.full(len(barcodes_all),-1,np.int64); row_of[manifest.cell_index.values]=np.arange(len(manifest))
mapped=row_of[_global_cross]
has_local=mapped>=0
donors_cross=mapped.copy()
# cells whose cross-type donor is not among the 1000 manifest cells get a fallback
# donor drawn from the other manifest cells of a different type
_manifest_types=manifest.cell_type.astype(str).values
for i in np.flatnonzero(~has_local):
    other=np.flatnonzero(_manifest_types!=_manifest_types[i])
    donors_cross[i]=other[np.random.default_rng(42+i).integers(len(other))]
assert (_manifest_types[donors_cross]!=_manifest_types).all()
cells=[]; coverage=[]
for row in manifest.itertuples():
    sl=slice(ptr[row.cell_index],ptr[row.cell_index+1]); local=np.asarray(gi[sl]); u=np.asarray(ub[sl]); s=np.asarray(sb[sl])
    valid=(mapping[local]>0)&(u>0)&(s>0)
    g=mapping[local[valid]].astype(np.int64); u=u[valid].astype(np.int64); s=s[valid].astype(np.int64)
    assert len(np.unique(g))==len(g)
    order=np.argsort(g); g,u,s=g[order],u[order],s[order]
    cells.append((g,u,s))
    coverage.append(dict(cell_id=row.cell_id,cache_pairs=len(local),eligible_pairs=len(g),panels=int(np.ceil(len(g)/24))))
coverage=pd.DataFrame(coverage); coverage.to_csv(out/'coverage.csv',index=False)
manifest.to_csv(out/'cell_manifest.csv',index=False)
total=int(coverage.eligible_pairs.sum())
config={**cfg,'epochs':args.epochs,'gpu':args.gpu,'eligible_pairs':total,'panel_size':24,
        'panel_rule':'vocab ID order; consecutive groups of 24; final group padded',
        'background_rule':'highest U+S bins among non-target eligible genes, up to 256 total valid genes',
        'shuffle_rule':'joint background U/S permutation, gene IDs fixed; per-cell/panel seed 42',
        'cross_type_rule':'cross_type_context uses a different-cell-type donor background (cross-sample preferred, seed 42); targets identical to matched_context',
        'gene_universe':'cache genes with vocab ID >0 and U/S bins >0; all evaluated exactly once per condition'}
(out/'run_config.json').write_text(json.dumps(config,indent=2))
def status(**kw):
    (out/'status.json').write_text(json.dumps({'timestamp':time.time(),**kw},indent=2))
status(status='preflight',eligible_pairs=total)
print('Eligible pairs',total,'panels',coverage.panels.sum(),flush=True)
torch.set_num_threads(4); torch.manual_seed(42)
ok,trace,reason=resource_preflight(args.gpu,31,10,minimum_free_mb=60000,maximum_mean_utilization=10)
trace.to_csv(out/'resource_preflight.csv',index=False)
if not ok:
    status(status='resource_blocked',reason=reason); raise RuntimeError(reason)
device=torch.device(f'cuda:{args.gpu}')
torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(device).total_memory,device)
projector,_=load_projector(Path(cfg['projector']['path']),128); projector.to(device).eval()
def panel_inputs(index):
    g,u,s=cells[index]; dg,du,ds=cells[int(manifest.iloc[index].donor_row)]
    xg,xu,xs=cells[int(donors_cross[index])]
    result=[]
    for start in range(0,len(g),24):
        t=np.arange(start,min(start+24,len(g))); mask=np.ones(len(g),bool); mask[t]=False
        rest=np.flatnonzero(mask); rest=rest[np.lexsort((g[rest],-(u[rest]+s[rest])))][:256-len(t)]
        arrays=[np.zeros(256,np.int64) for _ in range(3)]
        # reserve 24 slots for the protected panel, including padding in the final panel
        rest=rest[:232]
        for a,x in zip(arrays,[g,u,s]): a[:len(t)]=x[t]; a[24:24+len(rest)]=x[rest]
        donor_rest=np.flatnonzero(~np.isin(dg,g[t])); donor_rest=donor_rest[np.lexsort((dg[donor_rest],-(du[donor_rest]+ds[donor_rest])))][:232]
        matched=[a.copy() for a in arrays]
        for a,x in zip(matched,[dg,du,ds]): a[24:]=0; a[24:24+len(donor_rest)]=x[donor_rest]
        cross_rest=np.flatnonzero(~np.isin(xg,g[t])); cross_rest=cross_rest[np.lexsort((xg[cross_rest],-(xu[cross_rest]+xs[cross_rest])))][:232]
        cross=[a.copy() for a in arrays]
        for a,x in zip(cross,[xg,xu,xs]): a[24:]=0; a[24:24+len(cross_rest)]=x[cross_rest]
        shuffled=[a.copy() for a in arrays]; rng=np.random.default_rng(42+index*10000+start)
        order=rng.permutation(np.arange(24,24+len(rest)))
        shuffled[1][24:24+len(rest)]=arrays[1][order]; shuffled[2][24:24+len(rest)]=arrays[2][order]
        target=[a.copy() for a in arrays]
        for a in target: a[24:]=0
        for entry in [matched,cross,shuffled,target]:
            assert all(np.array_equal(a[:24],b[:24]) for a,b in zip(arrays,entry))
        result.append((t,dict(full=arrays,matched_context=matched,cross_type_context=cross,within_cell_shuffle=shuffled,target_panel_only=target)))
    return result
def encode(model,g,u,s,aligned):
    tg,tv,ts=interleave_tokens(g,u,s)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        h=model._encode(*(torch.as_tensor(a,device=device) for a in [tg,tv,ts]),return_all=True).float()
    with torch.inference_mode():
        hu,hs=h[:,0::2],h[:,1::2]
        raw=cosine_distance(hu,hs).cpu().numpy()
        align=cosine_distance(projector(hu),projector(hs)).cpu().numpy() if aligned else None
    return raw,align
conditions=['full','matched_context','cross_type_context','within_cell_shuffle','target_panel_only','pair_only']
def evaluate(model,index,aligned,wanted=None):
    wanted=conditions if wanted is None else wanted
    panels=panel_inputs(index); outputs={c:[] for c in wanted}; aligned_out={c:[] for c in wanted}
    for condition in wanted:
        if condition=='pair_only':
            g,u,s=[x[:,None] for x in cells[index]]
            for start in range(0,len(g),256):
                raw,z=encode(model,g[start:start+256],u[start:start+256],s[start:start+256],aligned)
                outputs[condition].extend(raw.ravel())
                if aligned: aligned_out[condition].extend(z.ravel())
        else:
            for start in range(0,len(panels),16):
                group=panels[start:start+16]
                arrays=[np.stack([p[1][condition][j] for p in group]) for j in range(3)]
                raw,z=encode(model,*arrays,aligned)
                for j,(t,_) in enumerate(group):
                    outputs[condition].extend(raw[j,:len(t)])
                    if aligned: aligned_out[condition].extend(z[j,:len(t)])
    records=[]
    for c in wanted:
        for space,values in [('raw',outputs[c])]+([('align',aligned_out[c])] if aligned else []):
            a=np.array(values); assert len(a)==len(cells[index][0]) and np.isfinite(a).all() and a.min()>=-1e-6 and a.max()<=2.000001
            records.append(dict(cell_id=manifest.iloc[index].cell_id,condition=c,space=space,n_genes=len(a),mean_distance=a.mean(),median_distance=np.median(a)))
    return pd.DataFrame(records)
def load(epoch):
    path=next(Path(cfg['checkpoint']).parent.glob(f'*epoch={epoch:02d}-*.ckpt'))
    return load_encoder({'data':{'vocab':str(vocab_path)},'model':{'checkpoint':str(path),'bin_size':15}},device)
model=load(11); started=time.time(); smoke=evaluate(model,0,True); repeated=evaluate(model,0,True)
assert np.array_equal(smoke.mean_distance,repeated.mean_distance)
smoke.to_csv(out/'smoke.csv',index=False)
seconds=(time.time()-started)/2
(out/'smoke_report.json').write_text(json.dumps({'passed':True,'seconds_per_cell':seconds,'estimated_serial_hours':seconds*5000/3600,'exact_repeat':True,'all_five_conditions':True},indent=2))
print('Smoke passed; seconds/cell',seconds,'estimated hours',seconds*5000/3600,flush=True)
del model; torch.cuda.empty_cache()
if args.smoke_only: status(status='smoke_passed'); raise SystemExit(0)
def run_cell(model,epoch,index,aligned,wanted,path):
    snap=query_gpu(args.gpu)
    if snap['memory_free_mb']<40000:
        status(status='resource_blocked',epoch=epoch,cell=index); raise RuntimeError('Memory pressure; resume supported')
    start=time.time(); result=evaluate(model,index,aligned,wanted); result['epoch']=epoch
    result.to_csv(path,index=False)
    status(status='running',epoch=epoch,cells_completed=index+1,seconds_last_cell=time.time()-start,eligible_pairs=total)
for epoch in map(int,args.epochs.split(',')):
    model=load(epoch); dest=out/f'epoch{epoch:02d}'; dest.mkdir(exist_ok=True)
    old_schema=(dest/f'cell_0000.csv').exists() and 'cross_type_context' not in pd.read_csv(dest/f'cell_0000.csv').condition.unique()
    for index in range(len(cells)):
        if old_schema:
            # legacy 5-condition file already covers this cell; only the new condition is missing
            xdest=dest/f'xtype_cell_{index:04d}.csv'
            if not xdest.exists():
                run_cell(model,epoch,index,epoch==11,['cross_type_context'],xdest)
        else:
            path=dest/f'cell_{index:04d}.csv'
            if path.exists(): continue
            run_cell(model,epoch,index,epoch==11,None,path)
        if index%50==0: print('epoch',epoch,'cells',index+1,'legacy' if old_schema else 'full',flush=True)
    del model; torch.cuda.empty_cache()
combined=pd.concat([pd.read_csv(p) for p in sorted(out.glob('epoch*/cell_*.csv'))]+[pd.read_csv(p) for p in sorted(out.glob('epoch*/xtype_cell_*.csv'))],ignore_index=True)
combined.to_csv(out/'cell_distances.csv',index=False)
assert combined.groupby(['epoch','space','condition']).size().eq(1000).all()
import runpy
runpy.run_path(str(Path(__file__).with_name('plot_all_gene_context_epochs.py')))
status(status='complete',eligible_pairs=total)

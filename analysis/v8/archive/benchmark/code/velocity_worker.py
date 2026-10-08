from common import *
import scanpy as sc,scvelo as scv
from scipy import sparse

def run(name,smoke=False):
 out=folder(name,smoke)
 if done(out,'velocity',name):return
 status(out,'velocity','preprocessing')
 np.random.seed(42);base=sc.read_h5ad(folder(name)/'data/input.h5ad')
 scv.pp.filter_and_normalize(base,min_shared_counts=protocol()['velocity_min_shared_counts'])
 sc.pp.log1p(base);sc.pp.highly_variable_genes(base,n_top_genes=min(protocol()['velocity_hvg'],base.n_vars),flavor='seurat');base=base[:,base.var.highly_variable].copy()
 if smoke:
  ids=pd.read_csv(out/'data/cells.csv',keep_default_na=False).cell_id.astype(str).to_numpy();positions=base.obs_names.get_indexer(ids);assert (positions>=0).all();base=base[positions].copy();assert np.array_equal(base.obs_names.astype(str),ids)
 sc.pp.pca(base,n_comps=min(30,base.n_obs-1,base.n_vars-1),random_state=42)
 sc.pp.neighbors(base,n_neighbors=min(30,base.n_obs-1),n_pcs=min(30,base.n_obs-1,base.n_vars-1),random_state=42)
 scv.pp.moments(base,n_pcs=min(30,base.n_obs-1,base.n_vars-1),n_neighbors=min(30,base.n_obs-1))
 files=[]
 for mode in ['stochastic','dynamical']:
  a=base.copy();p=out/'velocity'/mode;p.mkdir(parents=True,exist_ok=True);status(out,'velocity',mode)
  if mode=='dynamical':scv.tl.recover_dynamics(a,n_jobs=protocol()['velocity_n_jobs'])
  scv.tl.velocity(a,mode=mode);v=a.layers['velocity'];v=v.toarray() if sparse.issparse(v) else v
  np.save(p/'velocity_full.partial.npy',v);(p/'velocity_full.partial.npy').replace(p/'velocity_full.npy');np.save(p/'producer_mask.npy',a.var['velocity_genes'].to_numpy(bool));pd.DataFrame({'gene':a.var_names}).to_csv(p/'genes.csv',index=False);a.var.to_csv(p/'fit_parameters.csv')
  dump(p/'audit.json',dict(mode=mode,scvelo_version=scv.__version__,shape=list(v.shape),negative_entries=int((v<0).sum()),nonfinite_entries=int((~np.isfinite(v)).sum()),producer_genes=int(a.var.velocity_genes.sum()),input='raw U/S layers, filter_and_normalize then moments',cells_unchanged=bool(np.array_equal(a.obs_names,base.obs_names))))
  files.extend(p.iterdir())
 complete(out,'velocity',name,files);status(out,'velocity','complete')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.dataset,a.smoke)

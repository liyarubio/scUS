"""Reconstruct all-gene continuous moments in isolated output; no model inference."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']: os.environ[k]='4'
from pathlib import Path
import ast,json,hashlib,importlib.metadata
import numpy as np,pandas as pd,anndata as ad,scanpy as sc,scvelo as scv
import scipy.sparse as sp
OUT=Path(__file__).resolve().parents[1]
EXP=OUT.parent/'gene_distance_expression_decoupling/dataset_expansion_20260921/RAW2647LPS_GSE94383'
OLD=OUT.parent/'gene_distance_expression_decoupling/biological_evidence_rescreen_20260923/skap2_visualization_20260923'
cfg=json.loads((EXP/'source_run_config.json').read_text());run=Path(json.loads((EXP/'audit.json').read_text())['source_run'])
script=Path('/data1/liyaru/scripts/perturb_timecourse/timecourse_scus_s2_embedding_overview.py')
tree=ast.parse(script.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'read_concat_h5ads','apply_mouse_human_homologs'}]
exec(compile(ast.fix_missing_locations(tree),str(script),'exec'),globals())
np.random.seed(42)
a=read_concat_h5ads(cfg['h5ad']);a=apply_mouse_human_homologs(a,Path(cfg['mouse_human_homolog_map']),cfg['mouse_homolog_key'])
a.var_names_make_unique();a=a.copy();a.X=a.layers['spliced'].copy()
sc.pp.filter_cells(a,min_genes=100);sc.pp.filter_genes(a,min_cells=3)
scv.pp.filter_and_normalize(a);scv.pp.moments(a,use_highly_variable=False)
ids=np.load(run/'barcodes.npy',allow_pickle=True).astype(str);genes=np.load(run/'gene_names.npy',allow_pickle=True).astype(str)
assert np.array_equal(a.obs_names.astype(str).to_numpy(),ids)
ix=a.var_names.get_indexer(genes);assert (ix>=0).all()
mu=np.asarray(a.layers['Mu'][:,ix]);ms=np.asarray(a.layers['Ms'][:,ix])
ci=np.load(run/'frozen_pair_cache/cell_idx.int32.npy',mmap_mode='r');gi=np.load(run/'frozen_pair_cache/gene_idx.int32.npy',mmap_mode='r')
mask=np.zeros(mu.shape,bool);mask[ci,gi]=True
mismatch=int(np.count_nonzero(mask!=((mu>0)&(ms>0))));assert mismatch==0
old=pd.read_csv(OLD/'SKAP2_moments_all_cells.csv.gz').set_index('cell_id').loc[ids];j=list(genes).index('SKAP2')
err=max(float(np.max(np.abs(old.Mu.to_numpy()-mu[:,j]))),float(np.max(np.abs(old.Ms.to_numpy()-ms[:,j]))));assert err<1e-3
np.save(OUT/'data/Mu_reconstructed.npy',mu);np.save(OUT/'data/Ms_reconstructed.npy',ms)
from scvelo.preprocessing.neighbors import get_connectivities
con=get_connectivities(a,mode='connectivities',n_neighbors=30,recurse_neighbors=False);sp.save_npz(OUT/'data/moments_connectivities.npz',con)
meta=pd.read_csv(EXP/'cell_identity.csv.gz');labels=meta.timepoint_minutes.astype(str)+'_'+meta.stimulation
rows=[]
for lab in sorted(labels.unique()):
 w=np.asarray(con[labels.eq(lab).to_numpy()].mean(0)).ravel()
 rows.append({'source_group':lab,**{x:float(w[labels.eq(x)].sum()) for x in sorted(labels.unique())}})
pd.DataFrame(rows).to_csv(OUT/'audit/neighbor_group_mixing.csv',index=False)
record={'cells':len(ids),'genes':len(genes),'pair_support_mismatches':mismatch,'SKAP2_previous_reconstruction_max_abs_error':err,'scope':'new reconstruction using historical preprocessing; original continuous values not archived; no model re-encoding','all_839_cells_joint_graph':True,'versions':{k:importlib.metadata.version(k) for k in ['anndata','scanpy','scvelo','numpy','scipy']},'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}
(OUT/'audit/moments_reconstruction.json').write_text(json.dumps(record,indent=2));print(record,flush=True)

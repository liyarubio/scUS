from common import *
import h5py
from scipy import sparse
from anndata._io.specs import read_elem

def schema(path):
 with h5py.File(path) as f:return read_elem(f['obs']),read_elem(f['var'])
def read_rows(path,key,rows):
 rows=np.asarray(rows);sort=np.argsort(rows);ordered=rows[sort];chunks=[]
 with h5py.File(path) as f:
  x=f[key]
  if isinstance(x,h5py.Dataset):
   for start in range(0,len(rows),256):chunks.append(sparse.csr_matrix(x[ordered[start:start+256]]))
  else:
   if x.attrs['encoding-type']=='csc_matrix':return read_elem(x).tocsr()[rows]
   if x.attrs['encoding-type']!='csr_matrix':raise ValueError(f'Unsupported sparse encoding: {path}:{key}')
   ip=x['indptr'][:];ncols=int(x.attrs['shape'][1])
   for start in range(0,len(rows),256):
    ii=ordered[start:start+256];lo,hi=int(ii[0]),int(ii[-1])+1;left,right=int(ip[lo]),int(ip[hi]);z=sparse.csr_matrix((x['data'][left:right],x['indices'][left:right],ip[lo:hi+1]-left),shape=(hi-lo,ncols));chunks.append(z[ii-lo])
 return sparse.vstack(chunks,format='csr')[np.argsort(sort)]
def anchor_ids(name,cfg):
 if cfg.get('cohort_file'):
  f=pd.read_csv(cfg['cohort_file'],keep_default_na=False);return f.cell_id.to_numpy(),f.cell_type.to_numpy()
 if name=='forebrain':return pd.read_csv(cfg['anchor']).cell_id.astype(str).to_numpy(),None
 z=np.load(cfg['anchor'],allow_pickle=True)
 if name=='bm_hdmm':
  x=sparse.load_npz(cfg['distance']);genes=z['gene_counts']>=5000;keep=np.diff(x[:,genes].tocsr().indptr)>=1000
  return z['barcodes'][keep].astype(str),z['cell_types'][keep].astype(str)
 labels=z['cell_types' if 'cell_types' in z else 'celltypes'].astype(str)
 return z['barcodes'].astype(str),labels

def audit(name):
 out=folder(name);out.mkdir(parents=True,exist_ok=True);cfg=specs()[name];ids,labels=anchor_ids(name,cfg)
 assert len(ids)==cfg['n'] and len(set(ids))==len(ids),(name,len(ids))
 obs,var=schema(cfg['source']);key=pd.Index(obs.index.astype(str))
 if name=='oscc':
  lab,_=schema(cfg['label_source']);rows=pd.Index(lab.index.astype(str)).get_indexer(ids);assert (rows>=0).all();wanted=lab.Barcode.astype(str).iloc[rows].to_numpy();key=pd.Index(obs.Barcode.astype(str));source_rows=key.get_indexer(wanted);labels=lab[cfg['label']].astype(str).iloc[rows].to_numpy()
 else:source_rows=key.get_indexer(ids)
 assert key.is_unique and (source_rows>=0).all(),name
 source_labels=obs[cfg['label']].astype(str).iloc[source_rows].to_numpy()
 if labels is None:labels=source_labels
 if name!='oscc':assert np.array_equal(labels,source_labels),(name,'label mismatch')
 meta=pd.DataFrame({'cell_id':ids,'cell_type':labels,'source_row':source_rows,'source_cell_id':obs.index.astype(str)[source_rows]})
 for column in ['Sample','sample','obs_dataset','source','stage','Barcode','sample_id.x','predictions_unconstrained']:
  if column in obs:meta[column]=obs[column].astype(str).iloc[source_rows].to_numpy()
 if name=='forebrain':meta['sample']=[i.split(':')[0] for i in ids]
 (out/'data').mkdir(exist_ok=True);meta.to_csv(out/'data/cells.csv',index=False)
 var=var.copy();var.insert(0,'gene',var.index.astype(str));var.to_csv(out/'data/source_genes.csv',index=False)
 paths=[cfg['source'],cfg['anchor']]+([cfg['cohort_file']] if cfg.get('cohort_file') else [])+([cfg['label_source']] if name=='oscc' else [])+([cfg['distance']] if name=='bm_hdmm' else [])
 rec={'status':'passed','cells':len(meta),'genes':len(var),'input_hashes':{p:sha(p) for p in paths},'label_definition':cfg['label'],'cell_id_unique':True,'source_row_join_verified':True}
 dump(out/'data/audit.json',rec);print(name,len(meta),len(var),flush=True)
 return out

def complete_extreme_selection(name,indices,protected):
 full=folder(name)/'data';u=sparse.load_npz(full/'u_counts.npz');sp=sparse.load_npz(full/'s_counts.npz');x=u+sp;x.eliminate_zeros();genes=pd.read_csv(full/'genes.csv',keep_default_na=False).gene.astype(str)
 foundation_path=Path('/data1/liyaru/miniconda/envs/scFoundation/lib/python3.10/site-packages/omicverse/llm/scfoundation/OS_scRNA_gene_index.19264.tsv');fv=set(pd.read_csv(foundation_path,sep='\t').gene_name);robust=x.getnnz(axis=0)/x.shape[0]*100>=.05
 fl=x[:,genes.isin(fv).to_numpy()&robust].getnnz(axis=1)
 gv=read_json(Path(protocol()['scgpt_model'])/'vocab.json');gl=np.minimum(1200,x[:,genes.isin(gv).to_numpy()].getnnz(axis=1)+1)
 selected=set(map(int,indices));info={}
 for model,length in [('scFoundation',fl),('scGPT',gl)]:
  if length[list(selected)].max()<length.max():
   best=int(np.argmax(length));remove=max(selected-set(protected));selected.remove(remove);selected.add(best);protected.add(best)
  info[model]=dict(full_max=int(length.max()),smoke_max=int(length[list(selected)].max()))
 return np.sort(list(selected)),info

def prepare(name,smoke=False):
 full=folder(name);out=folder(name,smoke);out.mkdir(parents=True,exist_ok=True)
 if done(out,'prepare',name):return
 cfg=specs()[name]
 if not (full/'data/audit.json').exists():audit(name)
 cells=pd.read_csv(full/'data/cells.csv',keep_default_na=False)
 if smoke:
  # Selection uses mapped full-cohort moments/token lengths, not a reduced gene panel.
  lengths=np.load(full/'data/token_lengths.npy');coverage=np.load(full/'data/pair_counts.npy');rng=np.random.default_rng(42);chosen=set()
  for _,g in cells.groupby('cell_type',sort=True):chosen.add(int(g.index[0]))
  chosen.update(np.argsort(lengths,kind='stable')[-16:]);chosen.update(np.argsort(coverage,kind='stable')[:16])
  protected=set(chosen)
  remaining=np.setdiff1d(np.arange(len(cells)),list(chosen));chosen.update(rng.choice(remaining,min(256-len(chosen),len(remaining)),replace=False));indices,extremes=complete_extreme_selection(name,np.sort(list(chosen)),protected);cells=cells.iloc[indices].reset_index(drop=True)
 else:indices=np.arange(len(cells))
 data=out/'data';data.mkdir(exist_ok=True);cells.to_csv(data/'cells.csv',index=False)
 if smoke:
  import anndata as ad
  for side in ['u','s']:
   x=sparse.load_npz(full/f'data/{side}_counts.npz')[indices];sparse.save_npz(data/f'{side}_counts.npz',x)
  genes=pd.read_csv(full/'data/genes.csv');genes.to_csv(data/'genes.csv',index=False)
  import shutil
  shutil.copy2(full/'data/scus_genes.csv',data/'scus_genes.csv')
  shutil.copy2(full/'data/moments_protocol.json',data/'moments_protocol.json')
  for key in ['Mu','Ms','u_bins','s_bins','token_lengths','pair_counts']:np.save(data/f'{key}.npy',np.load(full/f'data/{key}.npy',mmap_mode='r')[indices])
  dump(data/'selection.json',dict(full_indices=indices.tolist(),selection='all labels + 16 longest scUS tokens + 16 lowest pair coverage + seed42 random; ensure longest native scFoundation/scGPT input via minimal deterministic swap; full-cohort moments',model_token_extremes=extremes))
 else:
  from scipy import sparse as sp
  obs,var=schema(cfg['source']);gene=var.index.astype(str).to_numpy();valid=np.asarray([g not in ('','nan','None') for g in gene]);assert valid.all()
  # Aggregate duplicate gene names, preserving first-seen order.
  uniq=pd.Index(pd.unique(gene));mapping=uniq.get_indexer(gene);aggregate=sp.csr_matrix((np.ones(len(gene)),(np.arange(len(gene)),mapping)),shape=(len(gene),len(uniq)))
  for side,layer in [('u','unspliced'),('s','spliced')]:
   status(out,'prepare','reading_counts',layer=layer)
   x=read_rows(cfg['source'],'layers/'+layer,cells.source_row.to_numpy());assert np.isfinite(x.data).all() and (x.data>=0).all() and np.allclose(x.data,np.rint(x.data),atol=1e-5)
   x=(x@aggregate).astype(np.float32).tocsr();x.eliminate_zeros();sparse.save_npz(data/f'{side}_counts.npz',x)
  pd.DataFrame({'gene':uniq}).to_csv(data/'genes.csv',index=False)
 u=sparse.load_npz(data/'u_counts.npz');s=sparse.load_npz(data/'s_counts.npz')
 import anndata as ad
 a=ad.AnnData(X=(u+s).tocsr(),obs=cells.set_index('cell_id'),var=pd.DataFrame(index=pd.read_csv(data/'genes.csv').gene.astype(str)));a.layers['unspliced']=u;a.layers['spliced']=s
 a.write_h5ad(data/'input.partial.h5ad');(data/'input.partial.h5ad').replace(data/'input.h5ad')
 if not smoke:make_moments(out,a)
 paths=[p for p in data.iterdir() if p.is_file() and p.suffix in ['.npy','.npz','.csv','.h5ad','.json']]
 complete(out,'prepare',name,paths,cells=len(cells));status(out,'prepare','complete',cells=len(cells))

def make_moments(out,a):
 import scanpy as sc
 from sklearn.neighbors import NearestNeighbors
 import sys
 sys.path.insert(0,str(ROOT/'code/vendor'))
 from scus.data.tokens import rank_bin
 data=out/'data';status(out,'prepare','moments')
 # Library-normalized U/S; one expression graph fitted on the complete target cohort.
 g=a.copy();sc.pp.normalize_total(g,target_sum=1e4);sc.pp.log1p(g);sc.pp.highly_variable_genes(g,n_top_genes=min(2000,g.n_vars),flavor='seurat');sc.pp.pca(g,n_comps=min(30,g.n_obs-1,int(g.var.highly_variable.sum())-1),mask_var='highly_variable',random_state=42)
 nn=NearestNeighbors(n_neighbors=min(30,a.n_obs),n_jobs=4).fit(g.obsm['X_pca']).kneighbors(g.obsm['X_pca'],return_distance=False)
 graph=sparse.csr_matrix((np.ones(nn.size)/nn.shape[1],(np.repeat(np.arange(a.n_obs),nn.shape[1]),nn.ravel())),shape=(a.n_obs,a.n_obs));sparse.save_npz(data/'moments_neighbors.npz',graph)
 vocab=read_json(protocol()['vocab']);vocab=vocab.get('gene2idx',vocab)
 keep=[i for i,gene in enumerate(a.var_names) if gene in vocab and int(vocab[gene])>0];assert keep
 ids=np.array([vocab[a.var_names[i]] for i in keep]);order=np.argsort(ids);keep=np.asarray(keep)[order];ids=ids[order]
 pd.DataFrame({'gene':a.var_names[keep],'vocab_id':ids}).to_csv(data/'scus_genes.csv',index=False)
 bins=[]
 for side,key in [('unspliced','Mu'),('spliced','Ms')]:
  x=a.layers[side];total=np.asarray(x.sum(1)).ravel();scale=np.divide(1e4,total,out=np.zeros_like(total,dtype=float),where=total>0);x=(sparse.diags(scale)@x).tocsr()
  x=x[:,keep].tocsr()
  m=np.lib.format.open_memmap(data/f'{key}.npy',mode='w+',dtype='float32',shape=(a.n_obs,len(keep)))
  for start in range(0,a.n_obs,256):m[start:start+256]=(graph[start:start+256]@x).toarray()
  m.flush()
 mu=np.load(data/'Mu.npy',mmap_mode='r');ms=np.load(data/'Ms.npy',mmap_mode='r')
 ub=np.lib.format.open_memmap(data/'u_bins.npy',mode='w+',dtype='uint8',shape=mu.shape);sb=np.lib.format.open_memmap(data/'s_bins.npy',mode='w+',dtype='uint8',shape=mu.shape)
 for i in range(a.n_obs):
  vals=np.empty(mu.shape[1]*2,np.float32);vals[::2]=np.log1p(mu[i]);vals[1::2]=np.log1p(ms[i]);b=rank_bin(vals);ub[i]=b[::2];sb[i]=b[1::2]
 ub.flush();sb.flush();np.save(data/'token_lengths.npy',2*((ub>0)|(sb>0)).sum(1));np.save(data/'pair_counts.npy',((ub>0)&(sb>0)).sum(1))
 dump(data/'moments_protocol.json',dict(input='per-modality library normalized to 10000',neighbors=30,graph='exact Euclidean kNN including self on U+S log-normalized HVG2000 PCA30',tokenization='joint interleaved log1p(Mu/Ms), rank-bin 15; zero retained as padding',fit_scope='all target cells',vocab_sha256=sha(protocol()['vocab'])))

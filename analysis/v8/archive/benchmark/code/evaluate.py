"""Full-cohort native metrics; fixed graph clustering; UMAP uses saved exact kNN."""
from common import *
from distances import distance_block,stream
from scipy import sparse
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score,normalized_mutual_info_score

def scores(labels,pred):return dict(ARI=adjusted_rand_score(labels,pred),AMI=adjusted_mutual_info_score(labels,pred),NMI=normalized_mutual_info_score(labels,pred))
def cluster(indices,labels,out):
 import igraph as ig,leidenalg as la
 n=len(indices);edges=np.column_stack((np.repeat(np.arange(n),indices.shape[1]),indices.ravel()));graph=ig.Graph(n=n,edges=edges.tolist(),directed=False);graph.simplify(multiple=True,loops=True);rows=[];preds={}
 for resolution in protocol()['leiden_resolutions']:
  for seed in protocol()['leiden_seeds']:
   part=la.find_partition(graph,la.RBConfigurationVertexPartition,resolution_parameter=resolution,seed=seed,n_iterations=-1);key=f'res{resolution}_seed{seed}';preds[key]=part.membership
   rows.append(dict(resolution=resolution,seed=seed,clusters=len(part),**scores(labels,part.membership)))
 frame=pd.DataFrame(rows);frame.to_csv(out/'sensitivity.csv',index=False);pd.DataFrame(preds).to_csv(out/'cluster_assignments.csv',index=False)
 return frame[(frame.resolution==1)&(frame.seed==42)].iloc[0].to_dict()

def coordinates(indices,dists,out):
 from umap.umap_ import fuzzy_simplicial_set,simplicial_set_embedding,find_ab_params
 from sklearn.utils import check_random_state
 # 15 non-self neighbors plus a structural self entry; no new neighbor search or UMAP-distance evaluation.
 n=len(indices);ix=np.column_stack((np.arange(n),indices)).astype(np.int32);dd=np.column_stack((np.zeros(n),dists)).astype(np.float32);rng=check_random_state(42);placeholder=np.zeros((n,1),np.float32)
 graph,_,_=fuzzy_simplicial_set(placeholder,ix.shape[1],rng,'euclidean',knn_indices=ix,knn_dists=dd)
 a,b=find_ab_params(1.,protocol()['umap_min_dist']);xy,_=simplicial_set_embedding(placeholder,graph,2,1.,a,b,1.,5,protocol()['umap_epochs'],'spectral',rng,'euclidean',{},False,{},False,parallel=False)
 assert xy.shape==(n,2) and np.isfinite(xy).all();np.save(out/'umap.npy',xy);sparse.save_npz(out/'umap_fuzzy_graph.npz',graph)
 dump(out/'umap_config.json',dict(nonself_neighbors=indices.shape[1],internal_knn_columns_including_self=ix.shape[1],min_dist=protocol()['umap_min_dist'],seed=42,epochs=protocol()['umap_epochs'],input='saved exact native-distance kNN',evaluated_on_umap=False))

def average_diagnostic(x,labels,metric,out,device):
 from sklearn.cluster import AgglomerativeClustering
 n=len(labels);rng=np.random.default_rng(42)
 if n<=5000:take=np.arange(n)
 else:
  _,inv=np.unique(labels,return_inverse=True);selected=[np.flatnonzero(inv==i)[0] for i in np.unique(inv)];rest=np.setdiff1d(np.arange(n),selected);take=np.sort(np.r_[selected,rng.choice(rest,5000-len(selected),replace=False)])
 xx=x[take];nsub=len(take);d=np.empty((nsub,nsub),np.float64)
 for q in range(0,nsub,256):
  for r in range(0,nsub,2048):d[q:q+256,r:r+2048]=distance_block(xx[q:q+256],xx[r:r+2048],metric,device)
 np.fill_diagonal(d,0);np.save(out/'average_subset_indices.npy',take)
 if not np.isfinite(d).all():dump(out/'average_diagnostic.json',dict(status='undefined',reason='No common observed features for some selected pairs'));return
 pred=AgglomerativeClustering(n_clusters=len(np.unique(labels[take])),metric='precomputed',linkage='average').fit_predict(d)
 dump(out/'average_diagnostic.json',dict(status='complete',cells=nsub,scope='all cells auxiliary' if n<=5000 else 'fixed subset auxiliary only',k_from_labels=True,**scores(labels[take],pred)))

def run(name,smoke=False,device='cuda'):
 if device=='cuda':
  import torch
  if not torch.cuda.is_available():raise RuntimeError('CUDA evaluation requested but driver is unavailable')
  torch.cuda.set_per_process_memory_fraction(min(.9,60*1024**3/torch.cuda.get_device_properties(0).total_memory))
  torch.backends.cuda.matmul.allow_tf32=False
 out=folder(name,smoke);meta=pd.read_csv(out/'data/cells.csv',keep_default_na=False);labels=meta.cell_type.to_numpy();manifest=pd.read_csv(out/'features/manifest.csv');records=[]
 for rep in manifest.itertuples(index=False):
  stage='evaluate_'+rep.representation;dest=out/'evaluation'/rep.representation;dest.mkdir(parents=True,exist_ok=True)
  if done(out,stage,name):records.append(read_json(dest/'metrics.json'));continue
  status(out,'evaluate',rep.representation)
  source=out/rep.path;identity=dict(feature_sha=sha(source),cells_sha=sha(out/'data/cells.csv'),signature=signature(name))
  if (dest/'identity.json').exists():assert read_json(dest/'identity.json')==identity,'Evaluation inputs/configuration changed'
  else:dump(dest/'identity.json',identity)
  x=sparse.load_npz(source) if source.suffix=='.npz' else np.load(source,mmap_mode='r');ix,dd,asw=stream(x,labels,rep.metric,out=dest/'distance_blocks',device=device,query=protocol()['block_rows'],reference=protocol()['block_cols'],k=15)
  np.save(dest/'knn_indices.npy',ix);np.save(dest/'knn_distances.npy',dd);pd.DataFrame({'cell_id':meta.cell_id,'cell_type':labels,'ASW':asw}).to_csv(dest/'cell_asw.csv',index=False)
  main=cluster(ix,labels,dest);coordinates(ix,dd,dest);average_diagnostic(x,labels,rep.metric,dest,device)
  record=dict(representation=rep.representation,cells=len(meta),features=rep.features,distance=rep.metric,ASW=float(asw.mean()) if np.isfinite(asw).all() else None,ASW_defined_cells=int(np.isfinite(asw).sum()),ASW_undefined_reason=None if np.isfinite(asw).all() else 'Native distances undefined for some pairs: no shared observed features',**main)
  dump(dest/'metrics.json',record);complete(out,stage,name,[p for p in dest.iterdir() if p.is_file()]);records.append(record)
 pd.DataFrame(records).to_csv(out/'metrics.csv',index=False);assert len(records)==29;status(out,'evaluate','complete')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda');args=p.parse_args();run(args.dataset,args.smoke,args.device)

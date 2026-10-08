"""29 native representations; cohort-only fitting, labels are not used."""
from common import *
from scipy import sparse
import sys,gc
sys.path.insert(0,str(ROOT/'code/vendor'))
from scus.iclr_distance import fit_distance_calibrator,apply_distance_calibrator
from transforms import standardize,count_inputs,velocity_features

def generate(name,smoke=False):
 out=folder(name,smoke)
 if done(out,'features',name):return
 status(out,'features','running')
 data=out/'data';enc=out/'encoding';dest=out/'features';dest.mkdir(exist_ok=True);n=len(pd.read_csv(data/'cells.csv'));manifest=[]
 def save(key,x,metric='rms',description='',columns=None):
  folder=dest/key;folder.mkdir(exist_ok=True)
  if metric=='rms':
   z,mean,sd=standardize(np.asarray(x));assert np.isfinite(mean).all();np.savez(folder/'standardization.npz',mean=mean,std=sd);x=z
  if sparse.issparse(x):
   assert np.isfinite(x.data).all();p=folder/'features.npz';sparse.save_npz(folder/'features.partial.npz',x);(folder/'features.partial.npz').replace(p)
  else:
   assert not np.isinf(x).any();p=folder/'features.npy';np.save(folder/'features.partial.npy',np.asarray(x,dtype=np.float32));(folder/'features.partial.npy').replace(p)
  if columns is not None:pd.DataFrame({'feature':columns}).to_csv(folder/'columns.csv',index=False)
  manifest.append(dict(representation=key,path=str(p.relative_to(out)),metric=metric,cells=n,features=x.shape[1],status='ready',description=description));print('features',key,x.shape,flush=True)
 ub=np.load(data/'u_bins.npy',mmap_mode='r');sb=np.load(data/'s_bins.npy',mmap_mode='r');mu=np.load(data/'Mu.npy',mmap_mode='r');ms=np.load(data/'Ms.npy',mmap_mode='r');raw=np.load(enc/'distance.npy',mmap_mode='r');genes=pd.read_csv(data/'scus_genes.csv').gene.to_numpy();pair=(ub>0)&(sb>0);assert np.array_equal(np.isfinite(raw),pair)
 threshold=protocol()['min_gene_observations'];panel=pair.sum(0)>=threshold;assert panel.any();pd.DataFrame({'gene':genes,'pair_observations':pair.sum(0),'eligible':panel}).to_csv(dest/'gene_panel.csv',index=False)
 for key,x,isbin in [('u_bin_profile',ub,True),('s_bin_profile',sb,True),('mu_profile',mu,False),('ms_profile',ms,False)]:
  eligible=(x>0).sum(0)>=threshold;y=np.asarray(x[:,eligible],np.float32)
  if isbin:y[y==0]=np.nan
  save(key,y,columns=genes[eligible]);del y
 rp=np.asarray(raw[:,panel]);bp=pair[:,panel]
 y=ub[:,panel].astype(np.float32)-sb[:,panel];y[~bp]=np.nan;save('direct_bin_difference',y,columns=genes[panel]);del y
 y=np.abs(mu[:,panel]-ms[:,panel]);y[~bp]=np.nan;save('continuous_moments_abs_difference',y,columns=genes[panel]);del y
 save('pair_mask',bp.astype(np.float32),'jaccard',columns=genes[panel])
 u=np.load(enc/'pooled_u.npy');s=np.load(enc/'pooled_s.npy')
 for key,y in [('u_only',u),('s_only',s),('concatenated_u_s',np.concatenate((u,s),1)),('state',(u+s)/2)]:save(key,y)
 hidden=[]
 for side in ['u','s']:
  h=np.load(enc/(side+'_sum.npy'),mmap_mode='r');eligible=np.isfinite(h).sum(0)>=threshold;y=np.asarray(h[:,eligible]);save(side+'_hidden_sum_profile',y,columns=genes[eligible]);hidden.append(y)
 save('concat_hidden_sum_profile',np.concatenate(hidden,axis=1));del hidden
 save('raw_distance_profile',rp,columns=genes[panel]);save('distance_scalar',np.nanmedian(rp,axis=1)[:,None])
 shuffled=rp.copy();rng=np.random.default_rng(42)
 for row in shuffled:rng.shuffle(row)
 save('gene_shuffled_distance',shuffled);del shuffled;gc.collect()
 status(out,'features','fitting_calibrator',cells=n,genes=int(panel.sum()))
 cal=fit_distance_calibrator(raw,ub,sb,np.arange(n),min_observations=threshold)
 np.savez(dest/'calibration.npz',gene_median=cal.gene_median,bin_residual=cal.bin_residual,gene_mad=cal.gene_mad,eligible=cal.eligible)
 calibrated=np.lib.format.open_memmap(dest/'calibrated.partial.npy',mode='w+',shape=raw.shape,dtype='float32')
 for start in range(0,n,256):calibrated[start:start+256]=apply_distance_calibrator(raw[start:start+256],ub[start:start+256],sb[start:start+256],cal)
 calibrated.flush();save('calibrated_distance_profile',calibrated[:,panel],columns=genes[panel]);del calibrated;(dest/'calibrated.partial.npy').unlink()
 for model in ['scGPT','scFoundation','scVI']:save(model,np.load(out/f'models/{model}/features.npy',mmap_mode='r'),'euclidean')
 u=sparse.load_npz(data/'u_counts.npz');s=sparse.load_npz(data/'s_counts.npz');g=pd.read_csv(data/'genes.csv').gene.to_numpy()
 for key,x in count_inputs(u,s).items():save('expression_'+key,x,'euclidean',columns=g)
 for mode in ['stochastic','dynamical']:
  p=out/'velocity'/mode;v=np.load(p/'velocity_full.npy',mmap_mode='r');mask=np.load(p/'producer_mask.npy');y,valid=velocity_features(v,mask)
  if y.shape[1]==0:raise ValueError(f'No usable {mode} velocity genes: inspect fitting diagnostics; do not mark software failure NA')
  save('scvelo_'+mode,y,'euclidean','signed cell×gene velocity',pd.read_csv(p/'genes.csv').gene.to_numpy()[valid]);np.save(p/'analysis_gene_mask.npy',valid)
 frame=pd.DataFrame(manifest);assert len(frame)==29 and frame.representation.is_unique;frame.to_csv(dest/'manifest.csv',index=False)
 complete(out,'features',name,list(dest.rglob('*.npy'))+list(dest.rglob('*.npz'))+list(dest.rglob('*.csv')));status(out,'features','complete')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();generate(a.dataset,a.smoke)

from distances import *
from sklearn.metrics import pairwise_distances,silhouette_samples
import tempfile

def oracle(x,metric):
 n=len(x);d=np.zeros((n,n))
 for i in range(n):
  for j in range(n):
   if metric=='rms':
    k=np.isfinite(x[i])&np.isfinite(x[j]);d[i,j]=np.sqrt(np.mean((x[i,k]-x[j,k])**2)) if k.any() else np.inf
   elif metric=='jaccard':
    union=np.logical_or(x[i],x[j]).sum();d[i,j]=1-np.logical_and(x[i],x[j]).sum()/union if union else 0
   else:d[i,j]=np.linalg.norm(x[i]-x[j])
 np.fill_diagonal(d,0);return d

def main(device='cpu'):
 rng=np.random.default_rng(42);base=rng.normal(size=(31,7));base[3]=base[2];labels=np.array(['a']*15+['b']*15+['singleton']);checks=[]
 for metric in ['euclidean','rms','jaccard']:
  x=base.copy()
  if metric=='rms':x[::3,::2]=np.nan;x[:,1]=0
  if metric=='jaccard':x=(x>0).astype(float);x[0]=0;x[1]=0
  d=oracle(x,metric);np.testing.assert_allclose(distance_block(x,x,metric,device),d,atol=2e-7)
  ix,ds,s=stream(x,labels,metric,device=device,query=7,reference=9,k=5)
  np.fill_diagonal(d,np.inf);expected=np.array([np.lexsort((np.arange(len(x)),row))[:5] for row in d]);np.testing.assert_array_equal(ix,expected);np.testing.assert_allclose(ds,np.take_along_axis(d,expected,1),atol=2e-7)
  np.fill_diagonal(d,0);np.testing.assert_allclose(s,silhouette_samples(d,labels,metric='precomputed'),atol=2e-7)
  with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as td:
   try:stream(x,labels,metric,out=td,device=device,query=7,reference=9,k=5,interrupt_after=14)
   except InterruptedError:pass
   ix2,ds2,s2=stream(x,labels,metric,out=td,device=device,query=7,reference=9,k=5);np.testing.assert_array_equal(ix,ix2);np.testing.assert_allclose(ds,ds2);np.testing.assert_allclose(s,s2)
  checks.append(metric+' full oracle, ties, self-exclusion, singleton ASW, recovery')
 x=np.array([[1,np.nan],[np.nan,1]],float);assert np.isinf(distance_block(x,x,'rms',device)[0,1]);assert distance_block(np.zeros((1,2)),np.zeros((1,2)),'jaccard',device)[0,0]==0
 dump(ROOT/f'tests/math_{device}.json',dict(passed=True,checks=checks+['no common features = infinity','empty masks Jaccard = 0'],device=device))
if __name__=='__main__':
 import sys
 main(sys.argv[1] if len(sys.argv)>1 else 'cpu')

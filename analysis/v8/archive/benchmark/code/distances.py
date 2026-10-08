"""Exact float64 native-distance streaming, deterministic tie-breaking, full ASW."""
from common import *
from scipy import sparse

def dense(x):return x.toarray() if sparse.issparse(x) else np.asarray(x)
def distance_block(a,b,metric,device='cpu'):
 a=np.asarray(dense(a),dtype=np.float64);b=np.asarray(dense(b),dtype=np.float64)
 if device=='cuda':
  import torch
  t=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float64)
  a=t(a);b=t(b)
  if metric=='rms':
   ma=torch.isfinite(a).double();mb=torch.isfinite(b).double();a=torch.nan_to_num(a);b=torch.nan_to_num(b);count=ma@mb.T
   d=torch.sqrt(torch.clamp((a*a)@mb.T+ma@(b*b).T-2*a@b.T,min=0)/count.clamp(min=1));d[count==0]=torch.inf
  elif metric=='jaccard':
   inter=a@b.T;union=a.sum(1)[:,None]+b.sum(1)[None,:]-inter;d=torch.where(union>0,1-inter/union.clamp(min=1),0)
  else:d=torch.sqrt(torch.clamp((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,min=0))
  return d.cpu().numpy()
 if metric=='rms':
  ma=np.isfinite(a).astype(float);mb=np.isfinite(b).astype(float);a=np.nan_to_num(a);b=np.nan_to_num(b);count=ma@mb.T
  with np.errstate(divide='ignore',invalid='ignore'):d=np.sqrt(np.maximum((a*a)@mb.T+ma@(b*b).T-2*a@b.T,0)/np.maximum(count,1))
  d[count==0]=np.inf
 elif metric=='jaccard':
  inter=a@b.T;union=a.sum(1)[:,None]+b.sum(1)[None,:]-inter;d=np.where(union>0,1-inter/np.maximum(union,1),0)
 else:d=np.sqrt(np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0))
 return d

def stream(x,labels,metric,out=None,device='cpu',query=256,reference=2048,k=15,interrupt_after=None):
 n=x.shape[0];k=min(k,n-1);labs,yi=np.unique(np.asarray(labels).astype(str),return_inverse=True);counts=np.bincount(yi);indicator=np.eye(len(labs))[yi]
 inds=np.full((n,k),-1,np.int64);dist=np.full((n,k),np.inf);asw=np.full(n,np.nan);start=0
 if out:
  out=Path(out);out.mkdir(parents=True,exist_ok=True)
  if (out/'progress.json').exists():
   p=read_json(out/'progress.json');assert p['shape']==list(x.shape) and p['metric']==metric
   start=p['next_row'];z=np.load(out/'partial.npz');inds[:start]=z['indices'];dist[:start]=z['distances'];asw[:start]=z['asw']
 for q in range(start,n,query):
  end=min(q+query,n);a=x[q:end];sums=np.zeros((end-q,len(labs)));invalid=np.zeros(end-q,bool);bestd=np.full((end-q,k),np.inf);besti=np.full((end-q,k),n,np.int64)
  for r in range(0,n,reference):
   stop=min(r+reference,n);d=distance_block(a,x[r:stop],metric,device)
   overlap=np.arange(max(q,r),min(end,stop));d[overlap-q,overlap-r]=0
   invalid|=(~np.isfinite(d)).any(1);sums+=np.where(np.isfinite(d),d,0)@indicator[r:stop]
   d[overlap-q,overlap-r]=np.inf
   candd=np.concatenate((bestd,d),axis=1);candi=np.concatenate((besti,np.broadcast_to(np.arange(r,stop),(end-q,stop-r))),axis=1)
   # lexsort gives exact distance-then-original-row order at all ties.
   order=np.lexsort((candi,candd),axis=1)[:,:k];bestd=np.take_along_axis(candd,order,1);besti=np.take_along_axis(candi,order,1)
  inds[q:end]=besti;dist[q:end]=bestd
  for i in range(end-q):
   label=yi[q+i]
   if invalid[i] or len(labs)<2:continue
   if counts[label]==1:asw[q+i]=0;continue
   aa=sums[i,label]/(counts[label]-1);means=sums[i]/counts;means[label]=np.inf;bb=means.min();asw[q+i]=(bb-aa)/max(aa,bb) if max(aa,bb)>0 else 0
  if out:
   temp=out/'partial.tmp.npz';np.savez(temp,indices=inds[:end],distances=dist[:end],asw=asw[:end]);temp.replace(out/'partial.npz');dump(out/'progress.json',dict(next_row=end,shape=list(x.shape),metric=metric))
  print(f'distance {metric} rows={end}/{n} finite_ASW={np.isfinite(asw[:end]).sum()}',flush=True)
  if interrupt_after is not None and end>=interrupt_after:raise InterruptedError('intentional recovery test')
 if (inds>=n).any() or not np.isfinite(dist).all():raise ValueError('Fewer than k finite off-diagonal neighbors; cannot construct requested graph')
 return inds,dist,asw

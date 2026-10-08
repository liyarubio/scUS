"""Shared value semantics. Missing tokens are distinct from observed zero moments."""
import numpy as np
from scipy import sparse

def count_inputs(u,s):
 assert u.shape==s.shape
 for x in [u,s]:
  a=x.data if sparse.issparse(x) else x
  if not np.isfinite(a).all() or (a<0).any() or not np.all(np.abs(a-np.rint(a))<=1e-5):raise ValueError('Expected finite nonnegative integer counts')
 result={}
 for key,x in [('u',u),('s',s),('u_plus_s',u+s)]:
  x=x.astype(np.float32).copy();result[key+'_counts']=x
  z=x.copy()
  if sparse.issparse(z):z.data=np.log1p(z.data)
  else:z=np.log1p(z)
  result[key+'_log1p']=z
 return result

def observed_tokens(u,s):
 pair=(u>0)&(s>0);uu=u.astype(float).copy();ss=s.astype(float).copy();uu[uu==0]=np.nan;ss[ss==0]=np.nan
 return uu,ss,pair

def standardize(x):
 mean=np.nanmean(x,axis=0);sd=np.nanstd(x,axis=0);safe=np.where(sd>0,sd,1)
 return ((x-mean)/safe).astype(np.float32),mean,sd

def pooled(u,s,pair):
 assert u.shape==s.shape and pair.shape==u.shape[:2]
 if not pair.any(1).all():raise ValueError('No paired gene for a cell')
 uu=np.sum(np.where(pair[:,:,None],u,0),axis=1)/pair.sum(1)[:,None]
 ss=np.sum(np.where(pair[:,:,None],s,0),axis=1)/pair.sum(1)[:,None]
 return {'u_only':uu,'s_only':ss,'concatenated_u_s':np.concatenate((uu,ss),axis=1),'state':(uu+ss)/2}

def velocity_features(v,producer_mask):
 valid=np.asarray(producer_mask,bool)&np.isfinite(v).all(0)&(np.nanstd(v,axis=0)>0)
 # Signed velocities, without log, absolute values, missing-to-zero, or PCA.
 return v[:,valid].copy(),valid

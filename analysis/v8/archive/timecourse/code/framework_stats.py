"""Reusable contrasts on independent rows, or explicitly exploratory cell rows.
Never infer replicate identity. Caller must retain the declared statistical unit.
"""
import numpy as np
from scipy.stats import t

def bh(p):
    p=np.asarray(p,float);shape=p.shape;p=np.where(np.isfinite(p),p,1).ravel();n=len(p)
    order=np.argsort(p,kind='stable');q=np.empty(n)
    q[order]=np.minimum(1,np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1])
    return q.reshape(shape)

def moments(x):
    x=np.asarray(x,float);ok=np.isfinite(x);n=ok.sum(0);v=np.where(ok,x,0)
    mean=np.divide(v.sum(0),n,out=np.full(x.shape[1],np.nan),where=n>0)
    # Two-pass variance avoids catastrophic cancellation.
    ss=np.where(ok,x-mean,0);ss=(ss*ss).sum(0)
    var=np.divide(ss,n-1,out=np.full(x.shape[1],np.nan),where=n>1)
    return n,mean,var

def contrast(a,b,scale='linear',margin=.05,pseudocount=.01):
    """b minus a. log_mean effect is log2 ratio of arithmetic means."""
    na,ma,va=moments(a);nb,mb,vb=moments(b)
    aa=va/np.maximum(na,1);bb=vb/np.maximum(nb,1)
    if scale=='log_mean':
        effect=np.log2((mb+pseudocount)/(ma+pseudocount))
        aa=aa/((ma+pseudocount)*np.log(2))**2;bb=bb/((mb+pseudocount)*np.log(2))**2
    else: effect=mb-ma
    se=np.sqrt(aa+bb)
    df=np.divide((aa+bb)**2,aa**2/np.maximum(na-1,1)+bb**2/np.maximum(nb-1,1),out=np.full_like(se,np.nan),where=(aa+bb)>0)
    valid=(na>=2)&(nb>=2)&np.isfinite(se)&(se>0)&np.isfinite(df)
    safe=np.where(valid,se,1);dof=np.where(valid,df,1)
    p0=2*t.sf(np.abs(effect)/safe,dof)
    pm=np.minimum(1,2*t.sf((np.abs(effect)-margin)/safe,dof))
    pe=np.maximum(t.sf((effect+margin)/safe,dof),t.cdf((effect-margin)/safe,dof))
    crit=t.ppf(.95,dof)
    return {'n0':na,'n1':nb,'mean0':ma,'mean1':mb,'effect':effect,'se':se,'df':df,
            'ci90_low':effect-crit*se,'ci90_high':effect+crit*se,
            'p_zero':np.where(valid,p0,1),'p_change':np.where(valid,pm,1),'p_equiv':np.where(valid,pe,1)}

def onset(changed):
    """Times x genes. 0/1/2 = sampled times; 3 = not detected, not a time."""
    changed=np.asarray(changed,bool)
    return np.where(changed.any(0),changed.argmax(0),3)

def categories(d,e):
    out=np.full(len(d),'neither_detected',object)
    out[(d<3)&(e==3)]='distance_only_through_300'
    out[(e<3)&(d==3)]='expression_only_through_300'
    out[(d<3)&(e<3)&(d<e)]='distance_first'
    out[(d<3)&(e<3)&(d>e)]='expression_first'
    out[(d<3)&(e<3)&(d==e)]='same_sampled_time'
    return out

def validate_biological_units(metadata,unit_col,condition_col,verified=False,minimum=3):
    """Guard for future replicate-level deployment; never invent units from cells."""
    if not verified or unit_col not in metadata or metadata[unit_col].isna().any():
        raise ValueError('Verified independent biological unit identities are required')
    counts=metadata.groupby(condition_col)[unit_col].nunique()
    if (counts<minimum).any(): raise ValueError('Insufficient independent units per condition')
    return counts

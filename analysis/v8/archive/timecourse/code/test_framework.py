import numpy as np,pandas as pd,pytest
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from framework_stats import bh,contrast,onset,categories,validate_biological_units

def test_bh_matches_reference():
    p=np.array([.001,.5,.03,1,.03,.9]);np.testing.assert_allclose(bh(p),multipletests(p,method='fdr_bh')[1])

def test_welch_matches_scipy():
    rng=np.random.default_rng(42);a=rng.normal(0,.2,(97,5));b=rng.normal(.1,.3,(64,5));a[:10,1]=np.nan
    r=contrast(a,b);ref=ttest_ind(a,b,axis=0,equal_var=False,nan_policy='omit')
    np.testing.assert_allclose(r['p_zero'],ref.pvalue,rtol=1e-10)
    np.testing.assert_allclose(r['df'],ref.df,rtol=1e-10)

def test_equivalence_not_nonsignificance():
    a=np.array([[-1.],[1.],[-1.],[1.]])*20;b=a+.001
    r=contrast(a,b,margin=.05)
    assert r['p_zero'][0]>.05 and r['p_equiv'][0]>.05
    a=np.tile([[-.001],[.001]],(100,1));b=a+.005;r=contrast(a,b,margin=.05)
    assert r['p_equiv'][0]<.05 and r['p_change'][0]>.05

def test_effect_threshold_and_tost_are_exclusive():
    rng=np.random.default_rng(1);a=rng.normal(1,.1,(150,20));b=rng.normal(1,.1,(200,20))+np.linspace(0,.2,20)
    r=contrast(a,b,margin=.05)
    assert not ((r['p_change']<.05)&(r['p_equiv']<.05)).any()
    np.testing.assert_array_equal(r['p_equiv']<.05,(r['ci90_low']>-.05)&(r['ci90_high']<.05))

def test_missing_distance_not_zero():
    a=np.array([[np.nan,0],[np.nan,.1],[np.nan,.2]])
    r=contrast(a,a+.1);assert r['n0'][0]==0 and r['n0'][1]==3
    assert np.isnan(r['effect'][0]) and r['p_change'][0]==1

def test_censoring_is_not_later_detection():
    D=np.array([[1,0,0,0],[0,0,1,0],[0,0,0,0]],bool)
    E=np.array([[0,1,0,0],[0,0,0,0],[1,0,0,0]],bool)
    assert list(categories(onset(D),onset(E)))==['distance_first','expression_only_through_300','distance_only_through_300','neither_detected']

def test_unverified_units_rejected():
    with pytest.raises(ValueError):validate_biological_units(pd.DataFrame({'cell':['a','b'],'condition':['A','B']}),'cell','condition',verified=False)

def test_log_ratio_direction_and_scaling():
    a=np.arange(1,101,dtype=float)[:,None];b=a*2
    r=contrast(a,b,scale='log_mean',margin=.25,pseudocount=0)
    np.testing.assert_allclose(r['effect'],1)
    rev=contrast(b,a,scale='log_mean',margin=.25,pseudocount=0)
    np.testing.assert_allclose(rev['effect'],-r['effect']);np.testing.assert_allclose(rev['p_change'],r['p_change'])

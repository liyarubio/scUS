from common import *
from transforms import *

def main():
 u=sparse.csr_matrix([[0,2,3],[1,0,1]],dtype=float);s=sparse.csr_matrix([[3,0,2],[2,4,0]],dtype=float);r=count_inputs(u,s)
 np.testing.assert_array_equal(r['u_plus_s_counts'].toarray(),u.toarray()+s.toarray())
 np.testing.assert_allclose(r['u_plus_s_log1p'].toarray(),np.log1p(u.toarray()+s.toarray()),rtol=1e-6)
 assert not np.allclose(r['u_plus_s_log1p'].toarray(),r['u_log1p'].toarray()+r['s_log1p'].toarray())
 uu,ss,pair=observed_tokens(u.toarray(),s.toarray());assert np.isnan(uu[0,0]) and pair.sum()==2
 moments=np.array([[0.,1],[2,1]]);z,mean,sd=standardize(moments);assert np.isfinite(z).all() and z[0,0]==-1 and np.all(z[:,1]==0)
 rng=np.random.default_rng(42);h1=rng.normal(size=(3,5,7));h2=rng.normal(size=(3,5,7));mask=rng.random((3,5))>.3;p=pooled(h1,h2,mask)
 assert p['concatenated_u_s'].shape==(3,14);np.testing.assert_allclose(p['state'],(p['u_only']+p['s_only'])/2)
 for i in range(3):np.testing.assert_allclose(p['u_only'][i],h1[i,mask[i]].mean(0))
 vel=np.array([[-2,1,np.nan,0],[1,-1,2,0],[3,2,3,0.]])
 result,keep=velocity_features(vel,[True,True,True,True]);np.testing.assert_array_equal(keep,[True,True,False,False]);assert result[0,0]==-2
 dump(ROOT/'tests/transforms.json',dict(passed=True,checks=['raw U+S addition before log1p','sparse input and real zeros preserved','padding bins become NaN','pair mask uses AND','constant feature standardization','pool genes not hidden coordinates','concat dimension','signed velocity retained','NaN or constant velocity genes excluded']))
if __name__=='__main__':main()

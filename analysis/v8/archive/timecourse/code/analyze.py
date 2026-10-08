import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import json,hashlib,warnings
import numpy as np,pandas as pd
from scipy import sparse
from framework_stats import bh,contrast,onset,categories
warnings.filterwarnings('ignore',category=RuntimeWarning)
OUT=Path(__file__).resolve().parents[1];EXP=OUT.parent/'gene_distance_expression_decoupling/dataset_expansion_20260921/RAW2647LPS_GSE94383'
TIMES=[75,150,300];P=json.loads((OUT/'protocol.json').read_text())
def savejson(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def log(x):print(x,flush=True)

def load():
    genes=np.load(EXP/'gene_names.npy',allow_pickle=True).astype(str);m=pd.read_csv(EXP/'cell_identity.csv.gz')
    cfg=json.loads((EXP/'audit.json').read_text());cache=Path(cfg['source_run'])/'frozen_pair_cache'
    ci=np.load(cache/'cell_idx.int32.npy');gi=np.load(cache/'gene_idx.int32.npy');dv=np.load(EXP/'raw_distance.float32.npy')
    key=ci.astype(np.int64)*len(genes)+gi;assert (np.diff(key)>0).all()
    d=np.full((len(m),len(genes)),np.nan);d[ci,gi]=dv;assert np.isfinite(d).sum()==len(dv)
    u=sparse.load_npz(EXP/'U_raw_counts.npz').toarray();s=sparse.load_npz(EXP/'S_raw_counts.npz').toarray();depth=np.load(EXP/'raw_US_library_depth.npy')
    uc=u/depth[:,None]*1e6;sc=s/depth[:,None]*1e6
    mp=pd.read_csv(EXP/'gene_mapping.csv.gz').set_index('gene').loc[genes];unambig=mp.mapping_unambiguous.to_numpy(bool)
    # Verify time/stimulation against independently saved provenance, by identity.
    prov=OUT.parent/'gene_distance_expression_decoupling/biological_evidence_rescreen_20260923/skap2_visualization_20260923/english_boxplots_20260924/source_audit_20260924/cell_provenance.csv'
    v=pd.read_csv(prov).set_index('cell_id').loc[m.cell_id]
    assert np.array_equal(v.minutes.to_numpy(),m.timepoint_minutes.to_numpy());assert np.array_equal(v.stimulation.to_numpy(),m.stimulation.to_numpy())
    m['group']=m.timepoint_minutes.astype(str)+'_'+m.stimulation;m['included_primary']=m.timepoint_minutes.eq(0)|m.stimulation.eq('LPS')
    m['unverified_name_prefix']=m.well_or_cell_id.str.extract(r'^(\d+)')[0]
    m.to_csv(OUT/'data/cells.csv.gz',index=False);mp.reset_index().to_csv(OUT/'data/genes.csv',index=False)
    pd.crosstab(m.unverified_name_prefix,m.group).to_csv(OUT/'audit/name_prefix_group_counts.csv')
    m.groupby(['timepoint_minutes','stimulation']).size().rename('cells').reset_index().to_csv(OUT/'tables/cohort.csv',index=False)
    arrays={'distance':d,'raw_total':uc+sc,'raw_U':uc,'raw_S':sc,'raw_total_matched':np.where(np.isfinite(d),uc+sc,np.nan)}
    mu=np.load(OUT/'data/Mu_reconstructed.npy');ms=np.load(OUT/'data/Ms_reconstructed.npy');arrays['moments_total']=mu+ms
    np.save(OUT/'data/distance.npy',d);np.save(OUT/'data/raw_total_CPM.npy',uc+sc)
    both=np.isfinite(d)&(u>0)&(s>0)
    arrays['distance_raw_positive']=np.where(both,d,np.nan);arrays['raw_total_raw_positive']=np.where(both,uc+sc,np.nan)
    n=np.isfinite(d).sum(1);sums=np.nansum(d,1)
    arrays['distance_centered']=d-(sums[:,None]-d)/(n[:,None]-1)
    # The center is a sensitivity readout, never replaces raw distance.
    paths=[EXP/x for x in ['gene_names.npy','cell_identity.csv.gz','gene_mapping.csv.gz','U_raw_counts.npz','S_raw_counts.npz','raw_US_library_depth.npy','raw_distance.float32.npy','source_run_config.json']]+[cache/'cell_idx.int32.npy',cache/'gene_idx.int32.npy',prov,OUT/'protocol.json',OUT/'data/Mu_reconstructed.npy',OUT/'data/Ms_reconstructed.npy']
    manifest=[]
    for p in paths:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
        manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    pd.DataFrame(manifest).to_csv(OUT/'audit/input_manifest.csv',index=False)
    savejson(OUT/'audit/data_validation.json',{'cells':len(m),'genes':len(genes),'pairs':len(dv),'primary_cells':int(m.included_primary.sum()),'excluded_unstimulated_150':int((~m.included_primary).sum()),'pairs_unique':True,'provenance_exact':True,'finite_distances_in_0_2':bool(((dv>=0)&(dv<=2)).all()),'mapping_unambiguous_genes':int(unambig.sum()),'biological_replicates':'unverified'})
    return genes,m,unambig,arrays

def evaluate(genes,m,unambig,arrays):
    frames=[];idx0=np.flatnonzero(m.timepoint_minutes.eq(0)&m.stimulation.eq('No stimulation'))
    for time in TIMES:
        idx=np.flatnonzero(m.timepoint_minutes.eq(time)&m.stimulation.eq('LPS'));row={'gene':genes,'time_min':time}
        for name,x in arrays.items():
            r=contrast(x[idx0],x[idx],scale='linear' if name.startswith('distance') else 'log_mean',margin=.05 if name.startswith('distance') else .25)
            row.update({name+'_'+k:v for k,v in r.items()})
        q=pd.DataFrame(row);q['coverage0']=q.distance_n0/len(idx0);q['coverage1']=q.distance_n1/len(idx)
        q['eligible']=unambig&(q.distance_n0>=20)&(q.distance_n1>=20)&(q.coverage0>=.2)&(q.coverage1>=.2)&((q.coverage1-q.coverage0).abs()<=.1)&(q.raw_total_mean0>=1)&(q.raw_total_mean1>=1)
        for side in ['raw_U','raw_S']:
            for p in ['p_zero','p_change','p_equiv']:
                q.loc[(q[side+'_mean0']<1)|(q[side+'_mean1']<1),side+'_'+p]=1
        for side in ['distance_raw_positive','raw_total_raw_positive']:
            for p in ['p_zero','p_change','p_equiv']:
                q.loc[(q[side+'_n0']<20)|(q[side+'_n1']<20),side+'_'+p]=1
        for name in arrays:
            for p in ['p_zero','p_change','p_equiv']:q.loc[~q.eligible,name+'_'+p]=1
        frames.append(q);log(f'contrast {time}: eligible {q.eligible.sum()}')
    q=pd.concat(frames,ignore_index=True)
    for name in arrays:
        for test in ['zero','change','equiv']:q[name+'_q_'+test]=bh(q[name+'_p_'+test])
    joint={'joint_total':['distance_p_change','raw_total_p_equiv'],
           'joint_matched':['distance_p_change','raw_total_p_equiv','raw_total_matched_p_equiv'],
           'joint_US':['distance_p_change','raw_total_p_equiv','raw_total_matched_p_equiv','raw_U_p_equiv','raw_S_p_equiv'],
           'joint_moments':['distance_p_change','moments_total_p_equiv']}
    for name,cols in joint.items():q[name+'_p']=q[cols].max(1);q[name+'_q']=bh(q[name+'_p'])
    q['D_changed']=q.distance_q_change<.05;q['E_changed']=q.raw_total_q_change<.05;q['E_equivalent']=q.raw_total_q_equiv<.05
    q['D_changed_conventional']=(q.distance_q_zero<.05)&q.distance_effect.abs().ge(.05)
    q['E_changed_conventional']=(q.raw_total_q_zero<.05)&q.raw_total_effect.abs().ge(.25)
    q['candidate_total']=q.joint_total_q<.05;q['candidate_matched']=q.joint_matched_q<.05;q['candidate_US']=q.joint_US_q<.05
    q['formal_biological_status']='not_tested';q.to_csv(OUT/'tables/all_gene_tests.csv.gz',index=False)
    q[q.candidate_total].sort_values(['time_min','joint_total_q','gene']).to_csv(OUT/'tables/candidates_total.csv',index=False)
    q[q.candidate_matched].sort_values(['time_min','joint_matched_q','gene']).to_csv(OUT/'tables/candidates_matched.csv',index=False)
    q[q.candidate_US].sort_values(['time_min','joint_US_q','gene']).to_csv(OUT/'tables/candidates_US.csv',index=False)
    q[q.gene.eq('SKAP2')].to_csv(OUT/'tables/SKAP2_tests.csv',index=False)
    # Every region gets an explicit set-membership file, not only positive examples.
    membership=q[['gene','time_min','eligible','D_changed','E_changed','E_equivalent','candidate_total','candidate_matched','candidate_US','D_changed_conventional','E_changed_conventional']].copy()
    membership['region']=np.select([~q.eligible,q.D_changed&q.E_changed,q.D_changed&~q.E_changed,q.E_changed&~q.D_changed],['ineligible','both','distance_only','expression_only'],default='neither')
    membership.to_csv(OUT/'tables/venn_membership.csv',index=False)
    return q

def summarize(q,genes):
    rows=[]
    for time,z in q.groupby('time_min'):
        d=z.D_changed;e=z.E_changed;u=z.eligible
        rows.append(dict(time_min=time,universe=int(u.sum()),distance_only=int((d&~e).sum()),both=int((d&e).sum()),expression_only=int((e&~d).sum()),neither=int((u&~d&~e).sum()),distance_changed=int(d.sum()),expression_changed=int(e.sum()),expression_equivalent=int(z.E_equivalent.sum()),distance_only_equiv=int((d&~e&z.E_equivalent).sum()),distance_only_inconclusive=int((d&~e&~z.E_equivalent).sum()),joint_total=int(z.candidate_total.sum()),joint_matched=int(z.candidate_matched.sum()),joint_US=int(z.candidate_US.sum()),conventional_D=int(z.D_changed_conventional.sum()),conventional_E=int(z.E_changed_conventional.sum())))
    pd.DataFrame(rows).to_csv(OUT/'tables/venn_counts.csv',index=False)
    def mat(col):return np.stack([q[q.time_min.eq(t)].set_index('gene').loc[genes,col].to_numpy() for t in TIMES])
    ok=mat('eligible').all(0);d=onset(mat('D_changed'));e=onset(mat('E_changed'));cat=categories(d,e);cat[~ok]='not_eligible_all_times'
    strong=np.zeros(len(genes),bool);strict=strong.copy()
    for i in range(3):
        strong|=(d==i)&(e<3)&(e>i)&mat('candidate_total')[i]
        strict|=(d==i)&(e<3)&(e>i)&mat('candidate_matched')[i]
    tab=pd.DataFrame({'gene':genes,'eligible_all_times':ok,'distance_first_detection_min':[TIMES[i] if i<3 else np.nan for i in d],'expression_first_detection_min':[TIMES[i] if i<3 else np.nan for i in e],'ordering':cat,'early_distance_equivalence':strong&ok,'early_distance_equivalence_matched':strict&ok})
    tab['observed_detection_gap_min']=tab.expression_first_detection_min-tab.distance_first_detection_min
    tab.to_csv(OUT/'tables/onset_by_gene.csv',index=False)
    tab.groupby('ordering').size().rename('genes').reset_index().to_csv(OUT/'tables/onset_counts.csv',index=False)
    tab[tab.early_distance_equivalence].to_csv(OUT/'tables/early_distance_candidates.csv',index=False)
    # Same-cohort descriptive enrichment, no independent-gene binomial/Fisher p-value.
    a=q[q.time_min.eq(75)].set_index('gene').loc[genes];later=mat('E_changed')[1:].any(0);base=ok&a.E_equivalent.to_numpy();assoc=[]
    for name,mask in [('D_changed_75',base&a.D_changed.to_numpy()),('D_not_changed_75',base&~a.D_changed.to_numpy())]:
        assoc.append({'group':name,'genes':int(mask.sum()),'later_E_changed':int((mask&later).sum()),'later_E_changed_fraction':float(later[mask].mean()) if mask.any() else None,'scope':'same-cohort descriptive, correlated genes, no independent validation'})
    pd.DataFrame(assoc).to_csv(OUT/'tables/early_later_association.csv',index=False)
    return tab

def threshold_grid(q):
    from scipy.stats import t
    rows=[]
    for em in [.1,.25,.5]:
        for dm in [.025,.05,.1]:
            pe=np.minimum(1,2*t.sf((q.raw_total_effect.abs()-em)/q.raw_total_se,q.raw_total_df));pd_=np.minimum(1,2*t.sf((q.distance_effect.abs()-dm)/q.distance_se,q.distance_df))
            eq=np.maximum(t.sf((q.raw_total_effect+em)/q.raw_total_se,q.raw_total_df),t.cdf((q.raw_total_effect-em)/q.raw_total_se,q.raw_total_df))
            pe=np.where(q.eligible,pe,1);pd_=np.where(q.eligible,pd_,1);eq=np.where(q.eligible,eq,1)
            E=bh(pe)<.05;D=bh(pd_)<.05;J=bh(np.maximum(pd_,eq))<.05
            ge=np.stack([E[q.time_min.eq(t)] for t in TIMES]);gd=np.stack([D[q.time_min.eq(t)] for t in TIMES]);ok=np.stack([q.loc[q.time_min.eq(t),'eligible'] for t in TIMES]).all(0);cat=categories(onset(gd),onset(ge))
            rows.append({'expression_margin':em,'distance_margin':dm,'gene_contrast_joint_candidates':int(J.sum()),'distance_first':int(((cat=='distance_first')&ok).sum()),'expression_first':int(((cat=='expression_first')&ok).sum()),'same_sampled_time':int(((cat=='same_sampled_time')&ok).sum())})
    pd.DataFrame(rows).to_csv(OUT/'tables/threshold_sensitivity.csv',index=False)

def resample(genes,m,unambig,arrays,q):
    rng=np.random.default_rng(P['subsampling']['seed']);B=P['subsampling']['repeats'];N=P['subsampling']['cells_per_group'];groups=[np.flatnonzero((m.timepoint_minutes==t)&((m.stimulation=='LPS') if t else (m.stimulation=='No stimulation'))) for t in [0]+TIMES]
    counts=[];gene_counts={k:np.zeros(len(genes),int) for k in ['evaluable','distance_first','expression_first','same_sampled_time','early_distance_equivalence']}
    for b in range(B):
        picks=[rng.choice(g,N,replace=False) for g in groups];ds=[];es=[];eqs=[];eligible=[]
        for idx in picks[1:]:
            d=contrast(arrays['distance'][picks[0]],arrays['distance'][idx]);e=contrast(arrays['raw_total'][picks[0]],arrays['raw_total'][idx],scale='log_mean',margin=.25)
            ok=unambig&(d['n0']>=20)&(d['n1']>=20)&(abs(d['n0']-d['n1'])<=10)&(e['mean0']>=1)&(e['mean1']>=1)
            ds.append(np.where(ok,d['p_change'],1));es.append(np.where(ok,e['p_change'],1));eqs.append(np.where(ok,e['p_equiv'],1));eligible.append(ok)
        D=bh(ds)<.05;E=bh(es)<.05;J=bh(np.maximum(ds,eqs))<.05;ok=np.all(eligible,axis=0);di=onset(D);ei=onset(E);cat=categories(di,ei)
        early=np.zeros(len(genes),bool)
        for i in range(3):early|=(di==i)&(ei>i)&(ei<3)&J[i]
        r={'draw':b,'cells_per_group':N,'eligible_all_times':int(ok.sum())}
        gene_counts['evaluable']+=ok
        for name in ['distance_first','expression_first','same_sampled_time','distance_only_through_300','expression_only_through_300','neither_detected']:
            mask=ok&(cat==name);r[name]=int(mask.sum())
            if name in gene_counts:gene_counts[name]+=mask
        gene_counts['early_distance_equivalence']+=ok&early;r['early_distance_equivalence']=int((ok&early).sum());counts.append(r)
        if (b+1)%20==0:log(f'equal-size subsampling {b+1}/{B}')
    pd.DataFrame(counts).to_csv(OUT/'tables/equal_size_subsampling_counts.csv',index=False)
    f=pd.DataFrame({'gene':genes,**{k+'_draws':v for k,v in gene_counts.items()}})
    for k in gene_counts:f[k+'_frequency']=gene_counts[k]/B
    f.to_csv(OUT/'tables/equal_size_gene_stability.csv',index=False)

def controls(m,arrays,genes,unambig):
    # Supplementary time-matched control has only 16 cells: retain estimates, not main eligibility.
    a=np.flatnonzero(m.timepoint_minutes.eq(150)&m.stimulation.eq('No stimulation'));b=np.flatnonzero(m.timepoint_minutes.eq(150)&m.stimulation.eq('LPS'))
    row={'gene':genes}
    for name in ['distance','raw_total','moments_total']:
        r=contrast(arrays[name][a],arrays[name][b],scale='linear' if name=='distance' else 'log_mean',margin=.05 if name=='distance' else .25)
        row.update({name+'_'+k:v for k,v in r.items()})
    tab=pd.DataFrame(row);tab['status']='supplementary_16_control_cells_below_main_minimum_no_main_discovery_claim';tab.to_csv(OUT/'tables/time_matched_150_control.csv.gz',index=False)
    # Reproduce visible-style rank test, using corrected groups and all-gene multiplicity.
    from scipy.stats import mannwhitneyu
    rows=[];a=np.flatnonzero(m.timepoint_minutes.eq(0));j=list(genes).index('SKAP2')
    for time in TIMES:
        b=np.flatnonzero(m.timepoint_minutes.eq(time)&m.stimulation.eq('LPS'))
        for name in ['moments_total','distance']:
            x=arrays[name][a,j];y=arrays[name][b,j];x=x[np.isfinite(x)];y=y[np.isfinite(y)]
            if name=='moments_total':x=np.log1p(x);y=np.log1p(y)
            p=mannwhitneyu(x,y,alternative='two-sided',method='asymptotic').pvalue
            rows.append({'gene':'SKAP2','time_min':time,'readout':name,'n0':len(x),'n1':len(y),'mean0':x.mean(),'mean1':y.mean(),'p_MWU':p})
    z=pd.DataFrame(rows);z['q_six_SKAP2_tests_descriptive']=bh(z.p_MWU);z.to_csv(OUT/'tables/SKAP2_rank_test_context.csv',index=False)

if __name__=='__main__':
    savejson(OUT/'status.json',{'status':'running','stage':'load'})
    genes,m,unambig,arrays=load();log('loaded and identity verified')
    q=evaluate(genes,m,unambig,arrays);on=summarize(q,genes);threshold_grid(q);controls(m,arrays,genes,unambig);resample(genes,m,unambig,arrays,q)
    savejson(OUT/'status.json',{'status':'analysis_complete','genes':len(genes),'gene_contrast_tests':len(q),'formal_biological_inference':'not_tested','figures_report':'pending'})
    log(pd.read_csv(OUT/'tables/venn_counts.csv').to_string(index=False));log(pd.read_csv(OUT/'tables/onset_counts.csv').to_string(index=False))

"""Package and summarize saved time-course statistics without fitting or model calls."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import mannwhitneyu
M=Path(__file__).resolve().parent;L=M/'lps_data';D=M/'revision_data';D.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=json.loads((M/'sources_metadata/lps_inputs.json').read_text());assert all(sha(M/r['path'])==r['sha256'] for r in inputs)
cells=pd.read_csv(L/'cells.csv.gz');original=pd.read_csv(M/'figure_data/skap2.csv.gz')
z=original.merge(cells[['cell_id','timepoint_minutes','stimulation','included_primary']],on='cell_id',validate='one_to_one');assert len(z)==839 and z.timepoint_minutes.notna().all()
z=z[z.included_primary].copy();assert len(z)==823 and z.groupby('minutes').size().to_dict()=={0:186,75:145,150:368,300:124}
assert (z.minutes==z.timepoint_minutes).all() and z[z.minutes>0].stimulation.eq('LPS').all() and z[z.minutes.eq(0)].stimulation.eq('No stimulation').all()
z[['cell_id','minutes','stimulation','distance','log1p_Mu_plus_Ms']].to_csv(D/'skap2_timecourse.csv',index=False)
q=pd.read_csv(L/'all_gene_tests.csv.gz');assert len(q)==32721 and q.gene.nunique()==10907 and not q.duplicated(['gene','time_min']).any()
selected=q[q.gene.isin(['SKAP2','RPL23','RPS8'])].sort_values(['gene','time_min']);selected.to_csv(D/'lps_gene_effects.csv',index=False)
# Verify the complete historical 12-test family against the actual SKAP2 cells.
# Keep saved q values; do not adjust only the six displayed tests.
rank=pd.read_csv(M/'evidence/skap2_corrected_tests_recomputed.csv')
pvalues=[]
for row in rank.itertuples():
 time=int(row.case_group.split('_')[1])
 pvalues.append(mannwhitneyu(z[z.minutes.eq(time)][row.variable],z[z.minutes.eq(0)][row.variable],alternative='two-sided',method='asymptotic',use_continuity=True).pvalue)
order_p=np.argsort(pvalues);adjusted=np.empty(len(rank))
adjusted[order_p]=np.minimum(1,np.minimum.accumulate((np.array(pvalues)[order_p]*len(rank)/np.arange(1,len(rank)+1))[::-1])[::-1])
assert len(rank)==12 and np.allclose(pvalues,rank.p_cell_level,rtol=1e-10,atol=1e-30)
assert np.allclose(adjusted,rank.q_BH_12_cell_level,rtol=1e-10,atol=1e-30)
rank['time_min']=rank.case_group.str.split('_').str[-1].astype(int)
rank['panel']=rank.variable.map({'log1p_Mu_plus_Ms':'A','distance':'B'})
rank['annotation']=rank.q_BH_12_cell_level.map(lambda p:'****' if p<1e-4 else '***' if p<1e-3 else '**' if p<.01 else '*' if p<.05 else 'ns')
rank.to_csv(D/'figure4_significance.csv',index=False)
# C/D retain the raw-total expression and distance definitions of the screen.
examples=pd.read_csv(L/'example_cells.csv.gz')
rpl=examples[examples.gene.eq('RPL23') & examples.included_primary].copy()
assert len(rpl)==823 and set(rpl.cell_id)==set(z.cell_id) and not rpl.cell_id.duplicated().any()
assert rpl.set_index('cell_id').timepoint_minutes.sort_index().equals(z.set_index('cell_id').minutes.sort_index())
rpl['minutes']=rpl.timepoint_minutes;rpl['log1p_raw_total_CPM']=np.log1p(rpl.raw_total_CPM)
rpl[['cell_id','minutes','stimulation','raw_total_CPM','log1p_raw_total_CPM','distance']].to_csv(D/'rpl23_timecourse.csv',index=False)
assert rpl.groupby('minutes').distance.count().to_dict()=={0:185,75:144,150:367,300:124}
for row in selected[selected.gene.eq('RPL23')].itertuples():
 for col,prefix in [('raw_total_CPM','raw_total'),('distance','distance')]:
  assert np.isclose(rpl[rpl.minutes.eq(0)][col].mean(),getattr(row,prefix+'_mean0'),atol=1e-12,rtol=1e-12)
  assert np.isclose(rpl[rpl.minutes.eq(row.time_min)][col].mean(),getattr(row,prefix+'_mean1'),atol=1e-12,rtol=1e-12)
# Additional RPL23 rank tests requested for Figure 5C/D. Define the full
# six-test display family (two readouts x three comparisons), not a discovery set.
rpl_tests=[]
for column,panel in [('log1p_raw_total_CPM','C'),('distance','D')]:
 reference=rpl[rpl.minutes.eq(0)][column].dropna().to_numpy()
 for time in [75,150,300]:
  case=rpl[rpl.minutes.eq(time)][column].dropna().to_numpy()
  result=mannwhitneyu(case,reference,alternative='two-sided',method='asymptotic',use_continuity=True)
  rpl_tests.append(dict(gene='RPL23',panel=panel,variable=column,time_min=time,reference_min=0,n_reference=len(reference),n_case=len(case),U_case=float(result.statistic),p_cell_level=float(result.pvalue)))
rpl_rank=pd.DataFrame(rpl_tests);pv=rpl_rank.p_cell_level.to_numpy();ix=np.argsort(pv);qv=np.empty(len(pv));qv[ix]=np.minimum(1,np.minimum.accumulate((pv[ix]*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1])
rpl_rank['q_BH_6_cell_level']=qv
rpl_rank['annotation']=rpl_rank.q_BH_6_cell_level.map(lambda p:'****' if p<1e-4 else '***' if p<1e-3 else '**' if p<.01 else '*' if p<.05 else 'ns')
rpl_rank.to_csv(D/'rpl23_rank_tests.csv',index=False)
# Monotonic log1p display must agree with the rank test on raw total CPM.
for t in [75,150,300]:
 raw_p=mannwhitneyu(rpl[rpl.minutes.eq(t)].raw_total_CPM,rpl[rpl.minutes.eq(0)].raw_total_CPM,alternative='two-sided',method='asymptotic',use_continuity=True).pvalue
 assert np.isclose(raw_p,rpl_rank[rpl_rank.panel.eq('C')&rpl_rank.time_min.eq(t)].p_cell_level.iloc[0],rtol=1e-12,atol=0)
shown_ab=rank[rank.panel.notna()].copy();shown_ab['gene']='SKAP2';shown_ab['q_cell_level']=shown_ab.q_BH_12_cell_level;shown_ab['family_size']=12
shown_cd=rpl_rank.copy();shown_cd['q_cell_level']=shown_cd.q_BH_6_cell_level;shown_cd['family_size']=6
combined=pd.concat([shown_ab,shown_cd],ignore_index=True)
combined[['gene','panel','variable','time_min','p_cell_level','q_cell_level','family_size','annotation']].to_csv(D/'figure5_significance.csv',index=False)
newreport=dict(status='PASS',gene='RPL23',unit='cell',reference_minutes=0,test='Two-sided asymptotic Mann-Whitney U with tie and continuity correction',adjustment='BH across all six RPL23 display tests (2 readouts x 3 times)',scope='Exploratory selected-gene display; no independent culture-level inference, expression-equivalence claim, or genome-wide discovery correction',annotations=rpl_rank.to_dict('records'),data_source='revision_data/rpl23_timecourse.csv',source_sha256=sha(D/'rpl23_timecourse.csv'))
(D/'rpl23_rank_validation.json').write_text(json.dumps(newreport,indent=2)+'\n')
sigreport=dict(status='PASS',test='Two-sided asymptotic Mann-Whitney U; tie and continuity correction',reference_minutes=0,unit='cell; exploratory, not independent culture replicate',adjustment='Benjamini-Hochberg over the original 12 tests: 4 readouts x 3 times',saved_p_and_q_verified=True,annotations=rank[rank.panel.notna()][['panel','time_min','p_cell_level','q_BH_12_cell_level','annotation']].to_dict('records'),rpl23_expression_counts=[186,145,368,124],rpl23_distance_counts=[185,144,367,124],missing_distance='Excluded from panel D; never imputed',sources=[{'path':str(p.relative_to(M)),'sha256':sha(p)} for p in [M/'evidence/skap2_corrected_tests_recomputed.csv',M/'figure_data/skap2.csv.gz',L/'cells.csv.gz',L/'example_cells.csv.gz']])
(D/'figure4_validation.json').write_text(json.dumps(sigreport,indent=2)+'\n')
counts=[]
for time,s in q.groupby('time_min'):
 d=s.distance_q_change.lt(.05);e=s.raw_total_q_change.lt(.05)
 counts.append(dict(time_min=int(time),eligible=int(s.eligible.sum()),D_only=int((d&~e).sum()),E_only=int((e&~d).sum()),both=int((d&e).sum()),joint_total=int(s.joint_total_q.lt(.05).sum()),joint_US=int(s.joint_US_q.lt(.05).sum())))
counts=pd.DataFrame(counts);counts.to_csv(D/'lps_detection_counts.csv',index=False)
saved=pd.read_csv(L/'venn_counts.csv');assert (counts.D_only.to_numpy()==saved.distance_only.to_numpy()).all()
cand=q[q.joint_total_q.lt(.05)].sort_values(['gene','time_min']);assert len(cand)==4 and set(cand.gene)=={'RPL23','RPS8'} and q.joint_US_q.lt(.05).sum()==0
cand.to_csv(D/'lps_all_total_candidates.csv',index=False)
# Check exact SKAP2 distances against the full saved statistical table.
for row in selected[selected.gene.eq('SKAP2')].itertuples():
 assert np.isclose(z[z.minutes.eq(0)].distance.mean(),row.distance_mean0,atol=1e-12)
 assert np.isclose(z[z.minutes.eq(row.time_min)].distance.mean(),row.distance_mean1,atol=1e-12)
summary=z.groupby('minutes').agg(cells=('cell_id','size'),mean_log_moments=('log1p_Mu_plus_Ms','mean'),mean_distance=('distance','mean')).reset_index();summary.to_csv(D/'lps_skap2_summary.csv',index=False)
# Recompute category counts from the saved per-gene onset assignments, with censoring retained.
on=pd.read_csv(L/'onset_by_gene.csv');common=on[on.eligible_all_times];assert len(common)==3713
order=common.ordering.value_counts().to_dict();assert order['distance_first']==57 and order['expression_first']==45
onset=pd.DataFrame([{'ordering':k,'genes':int(v)} for k,v in order.items()]);onset.to_csv(D/'lps_onset_counts.csv',index=False)
def table(caption,label,cols,head,rows):
 return r'\begin{table}[ht]\setlength{\abovecaptionskip}{10pt}\setlength{\belowcaptionskip}{10pt}\centering'+'\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\begin{tabular}{'+cols+r'}\toprule '+head+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n'
tex=r'''\subsection{Complete saved time-course summaries}
\label{app:lps_summaries}
Tables~\ref{tab:lps_counts}, \ref{tab:lps_candidates} and~\ref{tab:lps_order} retain the eligible sets, every total-expression candidate and every detection-order category. The source archive also includes all 32,721 gene/time statistics and the frozen screening protocol. Gene symbols follow the historical human-homolog mapping of this mouse collection.
'''
tex+=table('Gene counts at each LPS time versus baseline. D-only means no detected expression change, not expression equivalence. Joint columns require distance change and expression equivalence.','tab:lps_counts','rrrrrrr','Time (min) & Eligible & D-only & E-only & Both & Joint total & Joint U/S',[[int(r.time_min),f'{int(r.eligible):,}',int(r.D_only),int(r.E_only),int(r.both),int(r.joint_total),int(r.joint_US)] for r in counts.itertuples()])
tex+=table('All four total-expression-equivalent, distance-changing records. E is the raw total-expression log2 mean ratio; D is the mean distance difference. These are exploratory cell-level tests.','tab:lps_candidates','lrrrrr','Gene & Time (min) & E effect & D effect & Joint $q$ & U/S joint $q$',[[r.gene,int(r.time_min),f'{r.raw_total_effect:.4f}',f'{r.distance_effect:.4f}',f'{r.joint_total_q:.4f}',f'{r.joint_US_q:.4f}'] for r in cand.itertuples()])
names={'neither_detected':'Neither detected by 300 min','expression_only_through_300':'Only expression detected by 300 min','distance_only_through_300':'Only distance detected by 300 min','same_sampled_time':'First detected at the same sampled time','distance_first':'Distance detected first; both eventually detected','expression_first':'Expression detected first; both eventually detected'}
tex+=table('Detection order among 3,713 genes eligible at every comparison. Undetected changes are censored at 300 minutes; these categories do not identify biological onset.','tab:lps_order','lr','Category & Genes',[[names[k],f'{v:,}'] for k,v in order.items()])
(M/'timecourse_appendix_tables.tex').write_text(tex)
s=summary.set_index('minutes');r=cand[cand.gene.eq('RPL23')]
main=r'''We revisit the SKAP2 time course (Figure~\ref{fig:perturbation}A,B). Mean U/S distances are '''+'/'.join(f'{s.loc[t,"mean_distance"]:.4f}' for t in [0,75,150,300])+r''', while mean $\log(1+M_u+M_s)$ values are '''+'/'.join(f'{s.loc[t,"mean_log_moments"]:.4f}' for t in [0,75,150,300])+r''' at 0/75/150/300 minutes. These readouts exhibit different temporal patterns. Small changes in the smoothed expression display do not establish unchanged raw expression: none of the three SKAP2 contrasts passes the raw-total equivalence screen, and only the 300-minute contrast passes the distance minimum-effect screen.

The full screen identifies '''+str(len(cand))+r''' gene/time records from two genes, RPL23 and RPS8, meeting joint total-expression equivalence and distance-change criteria. RPL23 has raw-total log2 mean ratios of '''+'/'.join(f'{v:.4f}' for v in r.raw_total_effect)+r''' and distance changes of '''+'/'.join(f'{v:.4f}' for v in r.distance_effect)+r''' at 75/150/300 minutes; Figure~\ref{fig:perturbation}C,D shows the corresponding cell distributions. It illustrates how the readout can nominate candidates beyond total-abundance change. However, no candidate passes the stricter joint screen requiring U and S separately equivalent; the RPL23 U-component point estimate at 75 minutes is about $+1.086$ log2. Total-abundance equivalence therefore does not establish fixed local inputs or an expression-independent mechanism.
'''
(M/'timecourse_results.tex').write_text(main)
report=dict(status='PASS',saved_gene_time_rows=len(q),genes=int(q.gene.nunique()),corrected_cells=len(z),group_sizes=summary[['minutes','cells']].to_dict('records'),candidate_records=len(cand),candidate_genes=sorted(cand.gene.unique()),joint_US_records=int(q.joint_US_q.lt(.05).sum()),common_onset_genes=len(common),onset_counts={k:int(v) for k,v in order.items()},scope='Saved-result summaries and corrected metadata join only. No model calls or new statistical fitting; existing p/q retain their exploratory status.')
(D/'lps_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

"""Re-summarize frozen numerical evidence; no model calls, clustering or fitted predictors."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
M=Path(__file__).resolve().parent;E=M/'evidence';D=M/'revision_data';D.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
DATASETS=['forebrain','oscc','mouse_gastrulation'];DS_NAMES=['Forebrain','OSCC regions','Mouse gastrulation']
CORE=['raw_distance_profile','concatenated_u_s','u_only','s_only','distance_scalar','gene_shuffled_distance','direct_bin_difference','continuous_moments_abs_difference','pair_mask','u_bin_profile','s_bin_profile']
LABELS={'raw_distance_profile':'Distance profile','concatenated_u_s':'Cell embedding (U/S pooled)','u_only':'U-only pooled','s_only':'S-only pooled','distance_scalar':'Distance median','gene_shuffled_distance':'Gene shuffle','direct_bin_difference':'U-bin minus S-bin','continuous_moments_abs_difference':'Absolute moment difference','pair_mask':'Pair-availability mask','u_bin_profile':'U-bin profile','s_bin_profile':'S-bin profile','mu_profile':'Unspliced moment profile','ms_profile':'Spliced moment profile','state':'Mean of U/S pooled vectors','u_hidden_sum_profile':'U hidden-coordinate sum profile','s_hidden_sum_profile':'S hidden-coordinate sum profile','concat_hidden_sum_profile':'Concatenated hidden-sum profiles','calibrated_distance_profile':'Locally calibrated distance profile','scGPT':'scGPT','scFoundation':'scFoundation','scVI':'scVI (target-trained)','expression_u_counts':'U counts','expression_u_log1p':'log(1+U counts)','expression_s_counts':'S counts','expression_s_log1p':'log(1+S counts)','expression_u_plus_s_counts':'U+S counts','expression_u_plus_s_log1p':'log(1+U+S counts)','scvelo_stochastic':'scVelo stochastic velocity','scvelo_dynamical':'scVelo dynamical velocity'}
MODELS=['input_embedding','random_42','random_43','random_44','epoch_11'];MODEL_NAMES=['Input embeddings','Random 42','Random 43','Random 44','Pretrained']
pd.DataFrame({'representation':CORE,'label':[LABELS[x] for x in CORE]}).to_csv(D/'core_methods.csv',index=False)
primary=pd.read_csv(E/'primary_metrics_recomputed.csv');grid=pd.read_csv(E/'clustering_grid_recomputed.csv');ctx=pd.read_csv(E/'context_distance_64_contrasts_recomputed.csv')
assert len(primary)==87 and len(grid)==783 and len(ctx)==320
assert set(primary.representation)==set(LABELS)
assert not grid.duplicated(['dataset','representation','resolution','seed']).any()
assert grid.groupby(['dataset','representation']).size().eq(9).all()
core=grid[grid.representation.isin(CORE)].copy();assert len(core)==297
core.to_csv(D/'core_clustering.csv',index=False)
primary.to_csv(D/'all_29_primary.csv',index=False)
metric=lambda rep,ds,key='NMI':float(primary[primary.dataset.eq(ds)&primary.representation.eq(rep)].iloc[0][key])
seq=lambda rep,key='NMI':'/'.join(f'{metric(rep,ds,key):.4f}' for ds in DATASETS)
keys=['gene','u_bin','s_bin','type_a','type_b'];assert not ctx.duplicated(['model']+keys).any()
expected=set(map(tuple,ctx[ctx.model.eq('epoch_11')][keys].to_numpy()))
assert len(expected)==64
for model in MODELS:assert set(map(tuple,ctx[ctx.model.eq(model)][keys].to_numpy()))==expected
# Always derive summaries from all 320 rows, without filtering effect size or sign.
assert np.allclose(ctx.min_absolute_effect,np.minimum(abs(ctx.effect_sample_1),abs(ctx.effect_sample_2)),atol=1e-12,rtol=0)
ctx['nonzero_same_direction']=ctx.effect_sample_1*ctx.effect_sample_2>0
ctx.to_csv(D/'context_effects.csv',index=False)
summary=ctx.groupby('model',sort=True).agg(contrasts=('gene','size'),genes=('gene','nunique'),median_min_abs_effect=('min_absolute_effect','median'),direction_count=('nonzero_same_direction','sum')).reset_index()
summary['direction_fraction']=summary.direction_count/summary.contrasts
summary.loc[summary.model.eq('input_embedding'),['direction_fraction']]=np.nan
bygene=ctx.groupby(['model','gene'],sort=True).agg(strata=('min_absolute_effect','size'),median_min_abs_effect=('min_absolute_effect','median'),direction_count=('nonzero_same_direction','sum')).reset_index()
bygene.to_csv(D/'context_by_gene.csv',index=False)
balanced=bygene.groupby('model').median_min_abs_effect.median();summary['median_of_gene_medians']=summary.model.map(balanced)
summary.to_csv(D/'context_model_summary.csv',index=False)
trained=ctx[ctx.model.eq('epoch_11')][keys+['min_absolute_effect']].rename(columns={'min_absolute_effect':'pretrained_effect'})
parts=[]
for model in MODELS[1:4]:
 r=ctx[ctx.model.eq(model)][keys+['min_absolute_effect']].rename(columns={'min_absolute_effect':'random_effect'})
 j=trained.merge(r,on=keys,how='inner',validate='one_to_one');assert len(j)==64
 j['random_model']=model;j['delta_pretrained_minus_random']=j.pretrained_effect-j.random_effect;parts.append(j)
delta=pd.concat(parts,ignore_index=True);delta.to_csv(D/'context_paired_differences.csv',index=False)
gd=delta.groupby(['random_model','gene']).agg(strata=('delta_pretrained_minus_random','size'),median_paired_delta=('delta_pretrained_minus_random','median')).reset_index();gd.to_csv(D/'context_paired_differences_by_gene.csv',index=False)
rows=[]
for model in MODELS[1:4]:
 vals=delta[delta.random_model.eq(model)].delta_pretrained_minus_random;g=gd[gd.random_model.eq(model)].median_paired_delta
 rows.append(dict(random_model=model,contrasts=len(vals),genes=len(g),median_paired_delta=vals.median(),positive_contrasts=int((vals>0).sum()),negative_contrasts=int((vals<0).sum()),tied_contrasts=int((vals==0).sum()),median_of_gene_median_deltas=g.median(),positive_genes=int((g>0).sum()),negative_genes=int((g<0).sum())))
ds=pd.DataFrame(rows);ds.to_csv(D/'context_paired_summary.csv',index=False)
# Descriptive, paired-setting comparisons. Configurations are not independent runs.
p=core[core.representation.eq('raw_distance_profile')][['dataset','resolution','seed','NMI','ARI']]
wins=[]
for rep in CORE[1:]:
 j=p.merge(core[core.representation.eq(rep)],on=['dataset','resolution','seed'],suffixes=('_profile','_comparator'),validate='one_to_one')
 for dataset,z in j.groupby('dataset'):
  wins.append(dict(dataset=dataset,representation=rep,settings=len(z),NMI_positive=int((z.NMI_profile>z.NMI_comparator).sum()),ARI_positive=int((z.ARI_profile>z.ARI_comparator).sum()),NMI_delta_min=float((z.NMI_profile-z.NMI_comparator).min()),NMI_delta_max=float((z.NMI_profile-z.NMI_comparator).max())))
pd.DataFrame(wins).to_csv(D/'core_setting_comparisons.csv',index=False)
# TeX numerical text is generated so the paper cannot silently drift from tables.
main=r'''At resolution 1 and seed 42, distance-profile NMI is '''+seq('raw_distance_profile')+r''', compared with '''+seq('concatenated_u_s')+r''' for the pooled U/S cell embedding (forebrain/OSCC/mouse). The single-state pooled U and S vectors score '''+seq('u_only')+' and '+seq('s_only')+r''', respectively. The profile also exceeds the distance median ('''+seq('distance_scalar')+r''') and gene shuffle ('''+seq('gene_shuffled_distance')+r''').

The comparison changes when the gene axis is retained in simple inputs. Signed U/S-bin differences score '''+seq('direct_bin_difference')+r''', absolute moment differences '''+seq('continuous_moments_abs_difference')+r''', and the availability mask '''+seq('pair_mask')+r'''. Thus, the distance profile leads these controls in forebrain, while each exceeds it in OSCC and mouse. Its ARI is '''+seq('raw_distance_profile','ARI')+r'''; Figure~\ref{fig:controls} displays both metrics for all core readouts.
'''
(M/'revision_results_main.tex').write_text(main)
get=lambda model:summary[summary.model.eq(model)].iloc[0]
s=get('epoch_11');g=bygene[bygene.model.eq('epoch_11')]
context=r'''Across all 64 contrasts, the pretrained encoder has a median smaller absolute within-library effect of '''+f'{s.median_min_abs_effect:.4f}'+r''', compared with '''+'/'.join(f'{get(m).median_min_abs_effect:.4f}' for m in MODELS[1:4])+r''' for random seeds 42/43/44. Direction agrees in '''+str(int(s.direction_count))+r''' of 64 contrasts ('''+f'{100*s.direction_fraction:.1f}'+r'''\%) for the pretrained encoder and '''+'/'.join(str(int(get(m).direction_count)) for m in MODELS[1:4])+r''' of 64 for the random encoders. Input embeddings have zero effects and no defined direction.

To reduce the influence of genes with many eligible strata, we also compute the median effect within each gene and then the median across the nine genes. This equal-gene summary is '''+f'{s.median_of_gene_medians:.4f}'+r''' for the pretrained encoder and '''+'/'.join(f'{get(m).median_of_gene_medians:.4f}' for m in MODELS[1:4])+r''' for the random encoders. Paired pretrained-minus-random differences have positive within-gene medians for '''+'/'.join(str(int(ds[ds.random_model.eq(m)].iloc[0].positive_genes)) for m in MODELS[1:4])+r''' of nine genes, respectively (Appendix~\ref{app:context_full}). These are descriptive summaries of overlapping strata, not independent biological replications.
'''
(M/'revision_context_main.tex').write_text(context)
# All 29 representations, exact dimensions and both metrics, three tables.
tex=r'\section{Complete cohort-level comparison}'+'\n'+r'\label{app:all_results}'+'\n'+r'All tables use resolution 1 and seed 42. Dimensions are the actual evaluated feature counts; the unit is a cell except for OSCC spatial observations. Protocol families are specified in Appendix~\ref{app:readouts}. The complete nine-setting grid is supplied as numerical evidence. No representation is removed because of its ranking.'+'\n'
order=CORE+['mu_profile','ms_profile','state','u_hidden_sum_profile','s_hidden_sum_profile','concat_hidden_sum_profile','calibrated_distance_profile','scGPT','scFoundation','scVI','expression_u_counts','expression_u_log1p','expression_s_counts','expression_s_log1p','expression_u_plus_s_counts','expression_u_plus_s_log1p','scvelo_stochastic','scvelo_dynamical'];assert len(order)==29
for dataset,name in zip(DATASETS,DS_NAMES):
 tex+=(r'\begin{table}[!htp]' if dataset=='forebrain' else r'\begin{table}[p]')+r'\setlength{\abovecaptionskip}{10pt}\setlength{\belowcaptionskip}{10pt}'+'\n'+r'\centering'+'\n'+r'\caption{'+name+r': all 29 evaluated representations. Higher NMI/ARI indicate closer agreement with the source annotations; differences are descriptive.}\label{tab:all_'+dataset+'}\n'+r'\begin{tabular}{p{.52\linewidth}rrr}\toprule Representation & Dimensions & NMI & ARI\\\midrule'+'\n'
 for rep in order:
  x=primary[primary.dataset.eq(dataset)&primary.representation.eq(rep)].iloc[0];tex+=f'{LABELS[rep]} & {int(x.dimensions):,} & {x.NMI:.4f} & {x.ARI:.4f}'+r'\\'+'\n'
 tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
(M/'web_supplement/all_29_results.tex').write_text(tex)
# Gene weighting audit, without new tests or selection.
tex=r'''\section{Complete matched-token comparison}
\label{app:context_full}
All five models use the same historical token cache within this experiment. The 20-gene panel yields nine genes with eligible contrasts in both libraries, with at least ten cells of each compared type per library and stratum. All 64 contrasts are retained, including discordant directions. For contrast $j$, let $d_{j\ell}$ be the signed Cohen effect in library $\ell$. We summarize $m_j=\min(|d_{j1}|,|d_{j2}|)$ and separately record whether $d_{j1}d_{j2}>0$. Identical input embeddings have zero within-stratum variation; the saved denominator-zero convention gives effect zero, and direction is undefined. These repeated contrasts share cells and genes.

The primary effect summary is $\operatorname{median}_j m_j$. To give each gene equal weight, the second summary is $\operatorname{median}_g\operatorname{median}_{j:g(j)=g}m_j$. For a paired pretrained/random comparison, we first form $\Delta_j=m_j^{\mathrm{pretrained}}-m_j^{\mathrm{random}}$ on identical gene/bin/type keys; the gene-level paired summary is $\operatorname{median}_{j:g(j)=g}\Delta_j$. A median of paired differences need not equal the difference of two medians. No significance tests or confidence intervals treat strata as independent replicates. Table~\ref{tab:context_controls} compares model summaries; Table~\ref{tab:context_genes} retains every gene; Table~\ref{tab:context_paired_genes} reports the paired gene-level differences.

\begin{table}[ht]\setlength{\abovecaptionskip}{10pt}\setlength{\belowcaptionskip}{10pt}\centering
\caption{All eligible matched-token contrasts. Equal-gene weighting takes the median of nine per-gene medians. A dash denotes undefined direction for identical input embeddings.}\label{tab:context_controls}
\begin{tabular}{lrrr}\toprule Model & Median effect & Equal-gene effect & Direction agreement\\\midrule
'''
for model,name in zip(MODELS,MODEL_NAMES):
 x=get(model);v='---' if model=='input_embedding' else f'{int(x.direction_count)}/64';tex+=f'{name} & {x.median_min_abs_effect:.4f} & {x.median_of_gene_medians:.4f} & {v}'+r'\\'+'\n'
tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
tex+=r'''\begin{table}[ht]\setlength{\abovecaptionskip}{10pt}\setlength{\belowcaptionskip}{10pt}\centering
\caption{Per-gene median smaller absolute effect. All nine genes and all 64 contrasts are represented. Random columns correspond to seeds 42, 43 and 44.}\label{tab:context_genes}
\begin{tabular}{lrrrrrr}\toprule Gene & Strata & Input & R42 & R43 & R44 & Pretrained\\\midrule
'''
for gene in sorted(ctx.gene.unique()):
 x=bygene[bygene.gene.eq(gene)].set_index('model');tex+=f'{gene} & {int(x.loc["epoch_11","strata"])} & '+' & '.join(f'{x.loc[m,"median_min_abs_effect"]:.4f}' for m in MODELS)+r'\\'+'\n'
tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
tex+=r'''\begin{table}[ht]\setlength{\abovecaptionskip}{10pt}\setlength{\belowcaptionskip}{10pt}\centering
\caption{Within-gene medians of the paired pretrained-minus-random effect difference. No gene is selected by the sign of its result.}\label{tab:context_paired_genes}
\begin{tabular}{lrrr}\toprule Gene & Pretrained--R42 & Pretrained--R43 & Pretrained--R44\\\midrule
'''
for gene in sorted(ctx.gene.unique()):
 x=gd[gd.gene.eq(gene)].set_index('random_model');tex+=gene+' & '+' & '.join(f'{x.loc[m,"median_paired_delta"]:.4f}' for m in MODELS[1:4])+r'\\'+'\n'
tex+=r'\bottomrule\end{tabular}\end{table}'+'\n'
tex+=r'The full 192 paired differences and their 27 gene-level summaries accompany the source archive. The context cache is distinct from the audited clustering pipeline. The pretrained effect exceeds each random model under the reported aggregate summaries, but this does not establish a consistent gain for every gene, improved direction agreement, or a controlled clustering benefit.'+'\n'
(M/'revision_context_appendix.tex').write_text(tex)
# Corrected SKAP2 values retained; formatting edited separately; no new biological tests.
manifest=dict(inputs=[dict(path=str((E/n).relative_to(M)),sha256=sha(E/n)) for n in ['primary_metrics_recomputed.csv','clustering_grid_recomputed.csv','context_distance_64_contrasts_recomputed.csv']],core_methods=CORE,models=MODELS,context_contrasts=64,context_genes=9,paired_differences=192,gene_paired_summaries=27,scope='Descriptive summaries of saved outputs; no training, inference, clustering, predictor fitting or significance tests.',formulas={'effect':'min(abs(Cohen_d_library1), abs(Cohen_d_library2))','paired_delta':'pretrained_effect - random_effect on identical gene/U-bin/S-bin/type-pair keys','equal_gene_effect':'median across genes of within-gene median effects','gene_paired_delta':'within-gene median of paired contrast deltas'},model_summary=summary.replace({np.nan:None}).to_dict('records'),paired_summary=ds.to_dict('records'))
(D/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
print(summary.to_string(index=False));print(ds.to_string(index=False));print('Core metric rows:',len(core))

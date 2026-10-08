"""Whole-population expression and all-bin distributions for the Figure 4B genes."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/scus_expression_mpl')
import json, hashlib, re
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/data1/liyaru')
SRC=BASE/'scUS/outputs/paper/forebrain_gene_context_examples'
MOM=BASE/'scUS/outputs/paper/pretraining_ablations/forebrain_fullinput_rms_benchmark_with_expression/umap_visualizations/moments_extension'
GENES=['FOXG1','EMX1','BCL11B']
TYPES=['Radial Glia','Neuroblast','Immature Neuron','Neuron']
XTICK_LABELS={ct:ct.replace(' ', '\n') if ' ' in ct else ct+'\n' for ct in TYPES}
ORDER_TEXT='Radial Glia > Neuroblast > Immature Neuron > Neuron'
CT={'Immature Neuron':'#1f77b4','Neuroblast':'#ff7f0e','Neuron':'#2ca02c','Radial Glia':'#d62728'}
INK='#21364A';PINK='#B63875'
for name in ['data','figures','audit']:(ROOT/name).mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.titlecolor':INK,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
meta=pd.read_csv(SRC/'cells.csv');genes=pd.read_csv(SRC/'encoded_genes.csv');b=np.load(SRC/'encoded_gene_bins.npz');spot=pd.read_csv(SRC/'spotlight_manifest.csv').set_index('gene')
dist=np.load(SRC/'models/epoch_11/distance.npy',mmap_mode='r')
provenance=json.loads((MOM/'moments_provenance.json').read_text());RAW=Path(provenance['h5ad'])
m=np.load(MOM/'reconstructed_moments.npz');mu=m['Mu'];ms=m['Ms'];moment_genes=m['gene_ids'];moment_cells=m['cell_ids'];m.close()
assert meta.cell_id.is_unique and np.array_equal(meta.cell_id.to_numpy(),moment_cells)
assert len(meta)==1720
samples=sorted(meta['sample'].unique())
raw_values={};totals={}
with h5py.File(RAW,'r') as f:
    raw_cells=f['obs'][f['obs'].attrs['_index']].asstr()[:]
    raw_genes=f['var'][f['var'].attrs['_index']].asstr()[:]
    ri=pd.Index(raw_cells).get_indexer(meta.cell_id)
    assert (ri>=0).all() and len(set(ri))==len(meta)
    cats=f['obs/clusters/categories'].asstr()[:];codes=f['obs/clusters/codes'][:]
    assert np.array_equal(cats[codes[ri]],meta.cell_type.to_numpy())
    gi=[]
    for gene in GENES:
        ix=np.flatnonzero(raw_genes==gene);assert len(ix)==1;gi.append(int(ix[0]))
    for side,layer in [('u','unspliced'),('s','spliced')]:
        x=f['layers'][layer]
        a=csr_matrix((x['data'][:],x['indices'][:],x['indptr'][:]),shape=tuple(x.attrs['shape']))
        assert np.isfinite(a.data).all() and (a.data>=0).all() and (a.data==np.floor(a.data)).all()
        totals[side]=np.asarray(a.sum(axis=1)).ravel().astype(float)[ri]
        assert (totals[side]>0).all()
        raw_values[side]=a[ri,:][:,gi].toarray().astype(float)
        del a
frames=[]
for j,gene in enumerate(GENES):
    col=int(np.flatnonzero(genes.gene_name==gene)[0]);mj=int(np.flatnonzero(moment_genes==genes.vocab_id.iloc[col])[0]);s=spot.loc[gene]
    p=meta.copy();p['gene']=gene;p['u_bin']=b['u'][:,col];p['s_bin']=b['s'][:,col]
    for side in ['u','s']:
        p[f'{side}_counts']=raw_values[side][:,j]
        p[f'{side}_library_total']=totals[side]
        p[f'{side}_cp10k']=raw_values[side][:,j]*10000/totals[side]
        p[f'log1p_{side}_cp10k']=np.log1p(p[f'{side}_cp10k'])
        p[f'log1p_{side}_counts']=np.log1p(p[f'{side}_counts'])
    p['Mu']=mu[:,mj];p['Ms']=ms[:,mj]
    p['log1p_Mu']=np.log1p(p.Mu);p['log1p_Ms']=np.log1p(p.Ms)
    p['paired_model_input']=(p.u_bin>0)&(p.s_bin>0)
    p['us_cosine_distance']=np.asarray(dist[:,col],float)
    p['original_bin_match']=(p.u_bin==int(s.u_bin))&(p.s_bin==int(s.s_bin))
    p['original_B_cell']=p.original_bin_match&p.cell_type.isin([s.type_a,s.type_b])
    assert np.array_equal(p.u_bin==0,p.Mu==0) and np.array_equal(p.s_bin==0,p.Ms==0)
    assert np.array_equal(np.isfinite(p.us_cosine_distance),p.paired_model_input)
    assert np.isfinite(p[['u_counts','s_counts','Mu','Ms']]).all().all()
    frames.append(p)
df=pd.concat(frames,ignore_index=True)
df.to_csv(ROOT/'data/all_cells_expression.csv',index=False)
old=pd.read_csv(SRC/'spotlight_cell_values.csv');old=old[old.gene.isin(GENES)]
assert set(zip(df.loc[df.original_B_cell,'gene'],df.loc[df.original_B_cell,'cell_id']))==set(zip(old.gene,old.cell_id))
summary=[];binrows=[];ranges=[]
for (gene,sample,ct),p in df.groupby(['gene','sample','cell_type'],sort=False):
    r={'gene':gene,'sample':sample,'cell_type':ct,'n_all':len(p),'n_original_B':int(p.original_B_cell.sum()),'original_B_fraction':float(p.original_B_cell.mean()),'n_original_bins':int(p.original_bin_match.sum()),'n_paired_model_input':int(p.paired_model_input.sum()),'distinct_u_bins':p.u_bin.nunique(),'distinct_s_bins':p.s_bin.nunique(),'distinct_bin_pairs':len(p[['u_bin','s_bin']].drop_duplicates())}
    for var in ['u_counts','s_counts','u_cp10k','s_cp10k','Mu','Ms']:
        v=p[var];r.update({f'{var}_mean':float(v.mean()),f'{var}_median':float(v.median()),f'{var}_q25':float(v.quantile(.25)),f'{var}_q75':float(v.quantile(.75)),f'{var}_zero_fraction':float((v==0).mean())})
    summary.append(r)
    counts=np.zeros((16,16),int)
    np.add.at(counts,(p.s_bin.to_numpy(),p.u_bin.to_numpy()),1)
    assert counts.sum()==len(p)
    for sb in range(16):
        for ub in range(16):binrows.append(dict(gene=gene,sample=sample,cell_type=ct,u_bin=ub,s_bin=sb,n=int(counts[sb,ub]),denominator=len(p),percent=100*counts[sb,ub]/len(p)))
    chosen=p[p.original_B_cell]
    if len(chosen):
        for var in ['u_counts','s_counts','u_cp10k','s_cp10k','Mu','Ms']:
            v=chosen[var];ranges.append(dict(gene=gene,sample=sample,cell_type=ct,n=len(v),variable=var,minimum=v.min(),q25=v.quantile(.25),median=v.median(),q75=v.quantile(.75),maximum=v.max(),unique_values=v.nunique()))
summary=pd.DataFrame(summary);bindata=pd.DataFrame(binrows);range_table=pd.DataFrame(ranges)
for frame in [summary,bindata,range_table]:
    frame['_gene_order']=frame.gene.map({g:i for i,g in enumerate(GENES)})
    frame['_type_order']=frame.cell_type.map({ct:i for i,ct in enumerate(TYPES)})
    frame.sort_values(['_gene_order','sample','_type_order'],kind='stable',inplace=True)
    frame.drop(columns=['_gene_order','_type_order'],inplace=True)
    frame.reset_index(drop=True,inplace=True)
summary.to_csv(ROOT/'data/expression_summary_by_library_celltype.csv',index=False)
bindata.to_csv(ROOT/'data/joint_bin_frequencies_including_zero.csv',index=False)
range_table.to_csv(ROOT/'data/original_B_continuous_expression_ranges.csv',index=False)
assert (bindata.groupby(['gene','sample','cell_type']).n.sum().to_numpy()==summary.groupby(['gene','sample','cell_type']).n_all.first().to_numpy()).all()
assert np.allclose(bindata.groupby(['gene','sample','cell_type']).percent.sum(),100)
rng=np.random.default_rng(20260924)
plots=[];layout=[]

def save(fig,stem,book=None):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();heading=fig._suptitle.get_window_extent(renderer);footer=fig.texts[-1].get_window_extent(renderer)
    bad=[]
    for ax in fig.axes:
        if ax.get_title() and heading.overlaps(ax.title.get_window_extent(renderer)):bad.append('heading/title')
        for t in ax.get_xticklabels():
            if t.get_visible() and footer.overlaps(t.get_window_extent(renderer)):bad.append('footer/xtick')
    assert not bad,(stem,bad)
    layout.append({'figure':stem,'header_footer_overlaps':bad})
    for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'figures/{stem}.{ext}',dpi=160)
    if book is not None:book.savefig(fig)
    plots.append(stem);plt.close(fig)

def violin(ax,p,var,zero_var,ylim):
    labels=[]
    for x,ct in enumerate(TYPES):
        q=p[p.cell_type==ct];v=q[var].to_numpy();jit=x+rng.uniform(-.16,.16,len(v));mark=q.original_B_cell.to_numpy()
        if len(v)>1 and np.ptp(v)>0:
            vv=ax.violinplot(v,positions=[x],widths=.75,showextrema=False)
            for body in vv['bodies']:body.set_facecolor(CT[ct]);body.set_alpha(.26);body.set_edgecolor('none')
        ax.scatter(jit,v,c=CT[ct],s=5,alpha=.28,edgecolors='none',zorder=2)
        ax.boxplot(v,positions=[x],widths=.17,showfliers=False,patch_artist=True,boxprops={'facecolor':'white','edgecolor':INK,'linewidth':.8},medianprops={'color':INK},whiskerprops={'color':INK,'linewidth':.8},capprops={'color':INK,'linewidth':.8})
        ax.scatter(jit[mark],v[mark],s=12,facecolors='none',edgecolors=PINK,linewidths=.6,alpha=.75,zorder=4)
        labels.append(f'{XTICK_LABELS[ct]}\nn={len(v)}\n0: {(q[zero_var]==0).mean():.0%}')
    ax.set_xticks(range(4),labels,fontsize=8.5);ax.set_xlim(-.5,3.5);ax.set_ylim(*ylim);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)

with PdfPages(ROOT/'expression_and_bins_all_genes.pdf') as book:
    for gene in GENES:
        g=df[df.gene==gene];s=spot.loc[gene]
        metrics=[('log1p_u_cp10k','U counts: log1p(CP10K)','u_counts'),('log1p_s_cp10k','S counts: log1p(CP10K)','s_counts'),('log1p_Mu','Reconstructed log1p(Mu)','Mu'),('log1p_Ms','Reconstructed log1p(Ms)','Ms')]
        fig,axes=plt.subplots(2,4,figsize=(20,9),sharey='col')
        for j,(var,title,zero) in enumerate(metrics):
            ymax=max(float(g[var].max())*1.08,.1)
            for i,sample in enumerate(samples):
                p=g[g['sample']==sample];ax=axes[i,j];violin(ax,p,var,zero,(-ymax*.025,ymax));ax.set_title(title,fontsize=12)
                if j==0:ax.set_ylabel(f'Library {sample[-3:]}\nExpression (display scale)')
        fig.suptitle(f'{gene}: whole-population expression across all bins | all 1,720 cells',fontsize=19,y=.985)
        fig.text(.045,.025,'Colored points: all cells, including zeros. Magenta open circles: original Figure 4B cells. Box: median / IQR / 1.5 IQR whiskers.\nCounts: each modality normalized to 10,000 across all source genes, then log1p. Mu/Ms: archived reconstruction, then log1p.\nDevelopmental display order: Radial Glia > Neuroblast > Immature Neuron > Neuron. 0: zero fraction in the indicated counts or moments, before log1p.',fontsize=10)
        fig.subplots_adjust(left=.05,right=.985,bottom=.23,top=.89,wspace=.28,hspace=.70)
        save(fig,f'{gene}_expression_all_cells',book)

        # Raw unnormalized counts retained as an alternate view.
        fig,axes=plt.subplots(2,2,figsize=(12,8),sharey='col')
        for j,side in enumerate(['u','s']):
            var=f'log1p_{side}_counts';ymax=max(float(g[var].max())*1.08,.1)
            for i,sample in enumerate(samples):
                ax=axes[i,j];violin(ax,g[g['sample']==sample],var,f'{side}_counts',(-.025*ymax,ymax));ax.set_title(f'{side.upper()}: log1p(raw counts)')
                ax.set_ylabel(f'Library {sample[-3:]}')
        fig.suptitle(f'{gene}: original counts, all cells and all bins',fontsize=17,y=.985)
        fig.text(.055,.025,'No depth normalization or smoothing in this view; log1p is used only for display. Zeros are retained.\nMagenta open circles: original Figure 4B. Developmental display order: Radial Glia > Neuroblast > Immature Neuron > Neuron.',fontsize=9)
        fig.subplots_adjust(left=.07,right=.98,bottom=.23,top=.88,hspace=.80,wspace=.25)
        save(fig,f'{gene}_raw_counts')

        # Compact marginal bin distributions: each cell-type row sums to 100%.
        fig,axes=plt.subplots(2,2,figsize=(16,8.5))
        for i,sample in enumerate(samples):
            for j,side in enumerate(['u','s']):
                ax=axes[i,j];values=[];labels=[]
                for ct in TYPES:
                    q=g[(g['sample']==sample)&(g.cell_type==ct)]
                    values.append(np.bincount(q[f'{side}_bin'].to_numpy(),minlength=16)*100/len(q))
                    labels.append(f'{ct}\n(n={len(q)})')
                values=np.asarray(values)
                im=ax.pcolormesh(np.arange(17)-.5,np.arange(5)-.5,values,cmap='Blues',norm=PowerNorm(.5,vmin=0,vmax=100),shading='flat')
                for y in range(4):
                    for x in range(16):
                        if values[y,x]>0:
                            label='<1' if values[y,x]<1 else f'{values[y,x]:.0f}'
                            ax.text(x,y,label,ha='center',va='center',fontsize=8,color='white' if values[y,x]>45 else INK)
                target=int(s[f'{side}_bin'])
                ax.add_patch(Rectangle((target-.5,-.5),1,4,fill=False,edgecolor=PINK,linewidth=1.5))
                ax.set_ylim(3.5,-.5);ax.set_yticks(range(4),labels);ax.set_xticks(range(16));ax.set_xlabel(f'{side.upper()}-bin (0 = no positive input)')
                ax.set_title(f'Library {sample[-3:]} | {side.upper()}-bin marginal distribution')
        fig.suptitle(f'{gene}: all U-bin and S-bin distributions by cell type',fontsize=18,y=.985)
        fig.subplots_adjust(left=.14,right=.90,bottom=.20,top=.88,wspace=.52,hspace=.53)
        cax=fig.add_axes([.935,.25,.014,.5]);cb=fig.colorbar(im,cax=cax,ticks=[0,1,5,20,50,100]);cb.set_label('% within each cell-type row (sqrt colors)')
        fig.text(.055,.04,'Every row includes all cells and sums to 100% before rounding. Numbers are percentages; <1 denotes a positive fraction below 1%.\nMagenta outlines mark the original target bin for each modality separately, not the joint-bin subset. Zeros are retained.\nDevelopmental display order: Radial Glia > Neuroblast > Immature Neuron > Neuron.',fontsize=10)
        save(fig,f'{gene}_bin_marginals',book)

        # Exact joint bins, denominator is every cell in each library/type, including 0.
        fig,axes=plt.subplots(2,4,figsize=(17,10.5))
        for i,sample in enumerate(samples):
            for j,ct in enumerate(TYPES):
                p=g[(g['sample']==sample)&(g.cell_type==ct)];ax=axes[i,j]
                z=np.zeros((16,16),float);np.add.at(z,(p.s_bin.to_numpy(),p.u_bin.to_numpy()),100/len(p))
                z=np.ma.masked_equal(z,0);cmap=plt.colormaps['Blues'].copy();cmap.set_bad('#F5F6F8')
                im=ax.pcolormesh(np.arange(17)-.5,np.arange(17)-.5,z,cmap=cmap,norm=PowerNorm(gamma=.5,vmin=0,vmax=100),shading='flat',rasterized=False)
                selected_type=ct in [s.type_a,s.type_b]
                ax.add_patch(Rectangle((s.u_bin-.5,s.s_bin-.5),1,1,fill=False,edgecolor=PINK if selected_type else '#777777',linewidth=2,linestyle='-' if selected_type else '--'))
                ax.set_xlim(-.5,15.5);ax.set_ylim(-.5,15.5);ax.set_aspect('equal')
                ax.set_xticks([0,3,6,9,12,15]);ax.set_yticks([0,3,6,9,12,15]);ax.set_xlabel('U-bin (0 = no positive input)')
                ax.set_ylabel(f'Library {sample[-3:]} | S-bin' if j==0 else 'S-bin')
                hit=int(p.original_bin_match.sum());orig=int(p.original_B_cell.sum())
                ax.set_title(f'{ct} | all n={len(p)}\nOriginal bins: {hit} ({hit/len(p):.1%}); B cells: {orig}',fontsize=10.5,pad=10)
        fig.suptitle(f'{gene}: every observed U/S-bin combination, including zero',fontsize=18,y=.985)
        fig.subplots_adjust(left=.06,right=.90,bottom=.17,top=.87,wspace=.35,hspace=.38)
        cax=fig.add_axes([.93,.26,.015,.47]);cb=fig.colorbar(im,cax=cax,ticks=[0,1,5,20,50,100]);cb.set_label('% of all cells in this library / cell type\n(square-root color scale)',fontsize=10)
        fig.text(.055,.035,f'Each panel sums to 100%; empty tiles have zero cells. Original target bins: U={s.u_bin}, S={s.s_bin}.\nMagenta box: bin pair and cell types used in original B. Gray dashed box: same bins in other cell types.\nBin 0 = no positive corresponding model input; it is not raw-count zero. Developmental display order: Radial Glia > Neuroblast > Immature Neuron > Neuron.',fontsize=10)
        save(fig,f'{gene}_all_bin_pairs',book)

        # Continuous joint expression makes within-bin variation and selection visible.
        fig,axes=plt.subplots(2,4,figsize=(17,9.5),sharex=True,sharey=True)
        xmax=max(g.log1p_Mu.max()*1.08,.1);ymax=max(g.log1p_Ms.max()*1.08,.1)
        for i,sample in enumerate(samples):
            for j,ct in enumerate(TYPES):
                p=g[(g['sample']==sample)&(g.cell_type==ct)];ax=axes[i,j];selected=p[p.original_B_cell]
                ax.scatter(p.log1p_Mu,p.log1p_Ms,c=CT[ct],s=13,alpha=.5,edgecolors='none')
                ax.scatter(selected.log1p_Mu,selected.log1p_Ms,s=25,facecolors='none',edgecolors=PINK,linewidths=.9)
                ax.set_xlim(-xmax*.025,xmax);ax.set_ylim(-ymax*.025,ymax);ax.grid(alpha=.15)
                ax.set_title(f'{ct}\nn={len(p)}; original B={len(selected)}',fontsize=11)
                ax.set_xlabel('log1p(Mu)')
                if j==0:ax.set_ylabel(f'Library {sample[-3:]}\nlog1p(Ms)')
        fig.suptitle(f'{gene}: continuous U/S moments across all bins',fontsize=18,y=.985)
        fig.text(.055,.04,'Each point is a cell, including zero moments. Colored: all cells. Magenta rings: original Figure 4B subset.\nAxes are shared across all eight panels of this gene. Moments were reconstructed previously; bin agreement does not establish exact historical floats.\nDevelopmental display order: Radial Glia > Neuroblast > Immature Neuron > Neuron.',fontsize=10)
        fig.subplots_adjust(left=.06,right=.98,bottom=.19,top=.88,wspace=.20,hspace=.40)
        save(fig,f'{gene}_continuous_joint_expression',book)

coverage_rows=[]
for gene in GENES:
    q=summary[(summary.gene==gene)&(summary.n_original_B>0)]
    for r in q.itertuples():coverage_rows.append(f'| {gene} | {r.sample[-3:]} | {r.cell_type} | {r.n_all} | {r.n_original_B} | {r.original_B_fraction:.1%} |')
coverage_md='\n'.join(coverage_rows)
sections=''
for gene in GENES:
    sections+=f'''## {gene}

- [整体表达分布](figures/{gene}_expression_all_cells.png) · [PDF](figures/{gene}_expression_all_cells.pdf)
- [U/S 各自 bin 分布](figures/{gene}_bin_marginals.png) · [PDF](figures/{gene}_bin_marginals.pdf)
- [全部 bin 联合分布](figures/{gene}_all_bin_pairs.png) · [PDF](figures/{gene}_all_bin_pairs.pdf)
- [连续 Mu–Ms 联合分布](figures/{gene}_continuous_joint_expression.png) · [PDF](figures/{gene}_continuous_joint_expression.pdf)
- [原始 counts 分布（不做深度归一化）](figures/{gene}_raw_counts.png)

'''
report=f'''# Figure 4B 基因：整体表达和全部 bins 分布

本次按用户澄清后的问题，检查 FOXG1、EMX1、BCL11B 在全部细胞中的表达分布。**每个基因均保留同一批 1,720 个细胞、两个文库、四个细胞类型，包括不同 bins、bin 0、原始 counts 为 0 及缺失 U/S distance 的细胞。** 不再要求相同 bins，也没有每组至少 10 个细胞的筛选。

- [交互浏览入口](index.html)：切换基因与图种，图片可打开 SVG 放大。
- [全部主图 PDF（12 页）](expression_and_bins_all_genes.pdf)
- [逐细胞表达、bins 及原图选择标记](data/all_cells_expression.csv)
- [分文库/类型汇总](data/expression_summary_by_library_celltype.csv)

细胞类型全部使用全称，图中从左到右、热图及汇总表中从上到下，统一按 **Radial Glia → Neuroblast → Immature Neuron → Neuron** 排列。此顺序用于发育阶段展示，不代表本次分析推断了单细胞谱系。

## 三组图分别回答什么

1. **整体表达分布**：两行分别为两个文库；四列分别为 U counts、S counts、Mu、Ms；横轴是四种细胞类型。包含全部细胞和零值。U/S counts 各自按该模态全部源基因总 counts 归一化到 10,000 后作 log1p；Mu/Ms 作 log1p。颜色为细胞类型，粉色空心圈为原图 B 选中的细胞。箱线表示中位数、IQR、1.5 IQR 须；0% 是对应原始 counts 或 moments 的零值比例。另提供未经归一化的原始 counts 图（仅 log1p 展示）。
2. **完整 bin 联合分布**：每个文库 × 细胞类型一个面板；横轴 U-bin，纵轴 S-bin，包含 0 到 15 全部档位。颜色是该组**所有细胞**的百分比（平方根色标帮助显示低频组合），每个面板合计 100%。粉框标记原图 B 的 bin 和类型，灰虚框表示相同 bins 在其他类型中的覆盖。没有按配对 distance 是否可用来删细胞。
3. **连续 Mu–Ms 联合分布**：每个点一个细胞，横轴 log1p(Mu)、纵轴 log1p(Ms)，八个面板使用相同坐标范围。粉圈显示原图 B 是整体连续表达空间中的哪些细胞。

## 当前数据中直接可见的结果

- FOXG1：原图 B 仅覆盖对应文库/细胞类型全部细胞的约 12%–18%；EMX1 约 9%–13%，其余 bins 的细胞现在全部保留。
- BCL11B：两文库的 Immature Neuron 和 Neuron 均 100% 落在 U=15/S=15，因此在这两类细胞中不存在其他 bin 组合，但连续 Mu/Ms 仍有明显范围。其他细胞类型的不同 bins 也一并展示。
- FOXG1 原图同为 U=6/S=15 的细胞中，028 文库的 Ms 中位数约为 Immature Neuron 2.637、Radial Glia 1.209；029 为 2.864、1.235。这些是重建 moments 的描述性数值，直接说明相同 bins 不等于相同连续表达，不能把原图解释成表达等效比较。
- 原始 counts 与平滑 moments 呈现的零值比例不同；例如 FOXG1 各组原始 U counts 为 0 的比例约 92%–97%，需结合零值比例而不是仅看小提琴形状。

附加的 **U/S 各自 bin 分布图** 用每行百分比热图展开 0–15 全部档位，比只看联合分布更容易观察单个模态的分布；每行合计 100%（数字四舍五入可能有误差）。

## 原图 B 占整体比较群体多少

下表分母是该文库该细胞类型的全部细胞，含 bin 0 和无 distance 的细胞。其他细胞类型也保留在所有图中，原图 B 标记为 0。

| 基因 | 文库 | 细胞类型 | 全部细胞 | 原图 B 细胞 | 比例 |
|---|---|---|---:|---:|---:|
{coverage_md}

## 表达尺度与来源

- 原始 unspliced/spliced counts：`{RAW}`。逐 barcode 和唯一基因名匹配，细胞类型逐项一致；没有合并同名基因。
- counts 的 CP10K 为本次用于描述性比较的逐细胞、逐模态深度归一化，不是历史模型的预处理重放；原始 counts 和库总量保存在 CSV。
- Mu/Ms：`{MOM/'reconstructed_moments.npz'}`。这是**之前重建并保存的连续矩阵**，不是回收的历史原始浮点矩阵。原归档记录 1,720 个细胞的基因顺序和 bins 均一致（token bin agreement=1.0）；本次验证细胞顺序、目标基因 vocab ID、moments 零值与 bin 0、原图细胞集合。没有重新计算 moments。
- 历史分箱使用细胞内正值 log1p moments 的分位数；相同 bin 不对应跨细胞固定的绝对表达阈值，因此看整体表达必须结合连续值。bin 0 对这三个基因与对应 moments 为 0 一致，但不等于原始 counts 为 0。
- moments 在完整队列上构建并经邻域平滑；counts 的零值与 moments 的零值不能混用。表达图包含缺失 distance 的细胞，原有 distance NaN 仍保留。

## 如何解释

这些图用于检查完整表达分布及原图子集的覆盖偏倚。比较同一基因、同一表达尺度、两个文库中的类型间分布；不要把 U/S bins、counts、Mu/Ms 或 distance 的数值直接相互等同。原图匹配 bins 不代表连续表达完全一致；[原图子集的连续表达范围](data/original_B_continuous_expression_ranges.csv)给出具体范围。

没有新增独立样本、p/q 或表达等效性检验。细胞是展示单位，不是独立供者重复；分布不同或重叠都不能单凭此图作因果或表达独立性结论。

{sections}
## 审计和复现

- [完整联合 bin 计数（含零频格）](data/joint_bin_frequencies_including_zero.csv)
- [数据核验](audit/validation.json) · [源文件哈希](audit/source_manifest.json)
- [生成脚本](code/build_expression_overview.py)

```bash
/data1/liyaru/miniconda/bin/python {ROOT}/code/build_expression_overview.py
```
'''
(ROOT/'REPORT_CN.md').write_text(report)
summary_cols=['gene','sample','cell_type','n_all','n_original_B','original_B_fraction','u_counts_zero_fraction','s_counts_zero_fraction','Mu_median','Ms_median','distinct_bin_pairs']
table=summary[summary_cols].round(4).to_html(index=False,border=0)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Figure 4B 基因整体表达分布</title><style>body{font-family:system-ui,sans-serif;color:#21364a;max-width:1600px;margin:28px auto;padding:0 24px;line-height:1.65}h1{font-size:28px}a{color:#a72968}.note{background:#edf3f8;padding:16px 20px;border-radius:8px}.controls{position:sticky;top:0;background:#fff;z-index:2;padding:10px 0;border-bottom:1px solid #ddd}button,select{font:inherit;padding:8px 14px;margin:4px;border:1px solid #ccd4db;border-radius:6px;background:white;cursor:pointer}img{width:100%;height:auto;display:block;margin:20px 0}nav{display:flex;gap:20px;flex-wrap:wrap}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap}th,td{padding:8px;border-bottom:1px solid #dde2e8}th{background:#edf2f6}.warning{color:#8c3561}#caption{min-height:60px}</style>
<h1>FOXG1 / EMX1 / BCL11B：整体表达与全部 bins</h1><p>每个基因均纳入 1,720 个细胞、两个文库、四类细胞；包括不同 bins、bin 0、原始零表达和缺失 distance 的细胞。</p><nav><a href="REPORT_CN.md">完整中文说明</a><a href="expression_and_bins_all_genes.pdf">全部主图 PDF（12 页）</a><a href="data/all_cells_expression.csv">逐细胞表达 CSV</a><a href="data/expression_summary_by_library_celltype.csv">汇总 CSV</a></nav>
<div class="note"><strong>细胞类型顺序：Radial Glia → Neuroblast → Immature Neuron → Neuron。</strong>全称显示，按发育阶段排列。<br>先看“整体表达分布”，比较各类细胞的 U/S counts 与连续 Mu/Ms；再看“全部 bin 联合分布”，检查原图匹配 bins 覆盖了多少细胞。粉色空心圈或框标记原图 B 子集，其他细胞全部保留。<br>Mu/Ms 来自已保存的重建矩阵，历史 bins 核验一致；不是确认恢复的历史连续浮点值。原始 counts 单独保留。</div>
<div class="controls"><label>基因 <select id="gene"><option>FOXG1</option><option>EMX1</option><option>BCL11B</option></select></label><label>查看 <select id="view"><option value="expression_all_cells">整体表达分布</option><option value="bin_marginals">U/S 各自 bin 分布</option><option value="all_bin_pairs">全部 bin 联合分布</option><option value="continuous_joint_expression">连续 Mu–Ms 联合分布</option><option value="raw_counts">原始 counts（未归一化）</option></select></label><a id="pdf" href="figures/FOXG1_expression_all_cells.pdf">当前图 PDF</a> · <a id="svg" href="figures/FOXG1_expression_all_cells.svg">SVG 放大</a></div>
<p id="caption">两行分别是两个文库；四列是 U counts、S counts、Mu、Ms。横轴为四类细胞，包含全部细胞和零值。粉圈是原图 B 细胞。</p><a id="imageLink" href="figures/FOXG1_expression_all_cells.svg"><img id="chart" src="figures/FOXG1_expression_all_cells.png" alt="FOXG1 整体表达分布"></a>
<p class="warning">相同 bin ≠ 相同连续表达；bin 0 ≠ 原始 counts 为 0。本页是描述性表达检查，不是表达等效性检验或独立生物重复验证。</p>
<h2>所有基因 / 文库 / 细胞类型的覆盖与表达摘要</h2><p>n_all 为完整细胞数；n_original_B 为原图 B 子集；zero_fraction 为原始 counts 的零值比例；Mu/Ms_median 为未 log1p 的重建 moments 中位数。</p><div class="scroll">TABLE</div>
<h2>不使用切换控件时直接打开</h2>LINKS
<script>const gene=document.getElementById('gene'),view=document.getElementById('view');const texts={bin_marginals:'每行是一类细胞的全部细胞，U 和 S 分开展示所有 0–15 档位。格内数字为百分比，每行合计100%（显示值已四舍五入）；粉框仅标记原图相应模态的目标 bin，不代表两个 bins 同时匹配的子集。',expression_all_cells:'两行分别是两个文库；四列是 U counts、S counts、Mu、Ms。counts 各自作 CP10K 后 log1p，moments 作 log1p。细胞类型按 Radial Glia → Neuroblast → Immature Neuron → Neuron 排列，全部使用全称。0 为对应 counts 或 moments 的零值比例；粉圈是原图 B。',all_bin_pairs:'每个文库 × 细胞类型一个面板；横轴 U-bin，纵轴 S-bin，包括 0–15 全部档位。颜色分母为该组所有细胞，每面板合计100%，使用平方根色标显示低频格。粉框标记原图 B；灰虚框是相同 bins 在其他细胞类型的覆盖。',continuous_joint_expression:'横轴 log1p(Mu)，纵轴 log1p(Ms)，包括零值；同一基因八个面板坐标一致。每点一个细胞，粉圈标记原图 B 子集；可以观察相同 bins 内仍存在的连续表达差异。',raw_counts:'原始 unspliced / spliced counts 仅作 log1p 展示，未归一化、未平滑。保留全部细胞和零值；深度差异可能影响分布。粉圈是原图 B 子集。'};function render(){const stem=`figures/${gene.value}_${view.value}`;document.getElementById('chart').src=stem+'.png';document.getElementById('chart').alt=gene.value+' '+view.options[view.selectedIndex].text;document.getElementById('imageLink').href=stem+'.svg';document.getElementById('pdf').href=stem+'.pdf';document.getElementById('svg').href=stem+'.svg';document.getElementById('caption').textContent=texts[view.value]}gene.addEventListener('change',render);view.addEventListener('change',render);render();</script></html>'''
links=''
for gene in GENES:
    links+=f'<p><strong>{gene}</strong>：'+ ' · '.join(f'<a href="figures/{gene}_{suffix}.png">{label}</a>' for suffix,label in [('expression_all_cells','整体表达'),('bin_marginals','U/S各自bins'),('all_bin_pairs','联合bins'),('continuous_joint_expression','Mu–Ms散点'),('raw_counts','原始counts')])+'</p>'
(ROOT/'index.html').write_text(page.replace('TABLE',table).replace('LINKS',links))
inputs=[RAW,SRC/'cells.csv',SRC/'encoded_genes.csv',SRC/'encoded_gene_bins.npz',SRC/'spotlight_manifest.csv',SRC/'spotlight_cell_values.csv',MOM/'reconstructed_moments.npz',MOM/'moments_provenance.json',MOM/'token_reconstruction_audit.csv']
manifest=[]
for path in inputs:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    manifest.append({'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
assert manifest[0]['sha256']==provenance['h5ad_sha256']
(ROOT/'audit/source_manifest.json').write_text(json.dumps(manifest,indent=2))
validation={'status':'passed','n_cells_each_gene':1720,'genes':GENES,'cell_type_order':TYPES,'cell_type_labels':'full names','cell_gene_rows':len(df),'group_count':len(summary),'all_cells_retained_including_zero':True,'cell_ids_and_labels_exact':True,'raw_count_values_nonnegative_integers':True,'raw_gene_mapping_unique':True,'model_bin_zero_equals_moment_zero_for_target_genes':True,'distance_missingness_preserved':True,'original_B_cell_set_exact':True,'joint_bin_percent_per_group':100,'archived_full_token_bin_agreement':provenance['token_bin_agreement'],'historical_continuous_float_identity':'not established; cached reconstruction','moments_recomputed_this_run':False,'figures':plots,'header_footer_layout':layout,'browser_interaction_test':'not_run'}
(ROOT/'audit/validation.json').write_text(json.dumps(validation,indent=2))
print(json.dumps({k:v for k,v in validation.items() if k not in ['header_footer_layout','figures']},indent=2))
print(summary[['gene','sample','cell_type','n_all','n_original_B','u_counts_zero_fraction','s_counts_zero_fraction','Mu_median','Ms_median']].round(3).to_string(index=False))

# Keep the combined Mu-Ms plot as the first gallery view.
from plot_combined_mu_ms import build as build_combined_mu_ms
build_combined_mu_ms()

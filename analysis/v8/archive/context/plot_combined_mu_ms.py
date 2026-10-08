"""Overlay cell types in three Mu-Ms panels and make this the gallery's first plot."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/scus_expression_mpl')
import re,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
GENES=['FOXG1','EMX1','BCL11B']
TYPES=['Radial Glia','Neuroblast','Immature Neuron','Neuron']
COLORS={'Radial Glia':'#d62728','Neuroblast':'#ff7f0e','Immature Neuron':'#1f77b4','Neuron':'#2ca02c'}

def build():
    source=ROOT/'data/all_cells_expression.csv'
    df=pd.read_csv(source)
    assert not df.duplicated(['gene','cell_id']).any()
    assert set(df.cell_type)==set(TYPES)
    assert np.isfinite(df[['Mu','Ms']]).all().all() and (df[['Mu','Ms']]>=0).all().all()
    samples=sorted(df['sample'].unique());assert len(samples)==2
    assert df.groupby('gene').size().to_dict()=={g:1720 for g in GENES}
    assert all(set(df[df.gene==g].cell_id)==set(df[df.gene==GENES[0]].cell_id) for g in GENES)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#21364A','text.color':'#21364A','axes.titlecolor':'#21364A','pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    outputs=[]
    for scale in ['linear','log1p']:
        fig,axes=plt.subplots(1,3,figsize=(16,6.7))
        point_counts=[]
        for ax,gene in zip(axes,GENES):
            p=df[df.gene==gene].sample(frac=1,random_state=20260924)
            x=p.Mu.to_numpy();y=p.Ms.to_numpy()
            if scale=='log1p':x=np.log1p(x);y=np.log1p(y)
            ax.scatter(x,y,c=p.cell_type.map(COLORS).tolist(),marker='o',s=15,alpha=.72,edgecolors='white',linewidths=.12,rasterized=False)
            point_counts.append(sum(len(c.get_offsets()) for c in ax.collections))
            xmax=max(float(x.max()),.01);ymax=max(float(y.max()),.01)
            ax.set_xlim(-.025*xmax,1.06*xmax);ax.set_ylim(-.025*ymax,1.06*ymax)
            ax.set_xlabel('Mu' if scale=='linear' else 'log1p(Mu)')
            ax.set_ylabel('Ms' if scale=='linear' else 'log1p(Ms)')
            ax.set_title(f'{gene}\nn = {len(p):,} cells',fontsize=15,pad=12)
            ax.grid(alpha=.16);ax.set_axisbelow(True)
        assert point_counts==[1720]*3
        fig.suptitle('Mu-Ms expression: all cell types overlaid for each gene',fontsize=19,y=.975)
        handles=[Line2D([],[],marker='o',ls='',color=COLORS[ct],markersize=8,label=ct) for ct in TYPES]
        leg1=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,frameon=False,fontsize=11,handletextpad=.5,columnspacing=2)
        foot='Same 1,720 cells in each gene panel; all bins and zero moments retained. Axes are scaled separately for each gene.\nMu/Ms use the archived reconstructed moments; cell-type order follows Radial Glia > Neuroblast > Immature Neuron > Neuron.'
        fig.text(.055,.025,foot,fontsize=9.5)
        fig.subplots_adjust(left=.06,right=.985,bottom=.24,top=.73,wspace=.30)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        assert all(not leg1.get_window_extent(renderer).overlaps(ax.title.get_window_extent(renderer)) for ax in axes)
        stem=f'three_genes_Mu_Ms_celltype_{scale}'
        for ext in ['png','pdf','svg']:
            path=ROOT/f'figures/{stem}.{ext}';fig.savefig(path,dpi=190);outputs.append(str(path.relative_to(ROOT)))
        plt.close(fig)
    counts=df.groupby(['gene','sample','cell_type']).size().rename('n').reset_index()
    counts['_gene']=counts.gene.map({g:i for i,g in enumerate(GENES)});counts['_type']=counts.cell_type.map({t:i for i,t in enumerate(TYPES)})
    counts.sort_values(['_gene','sample','_type']).drop(columns=['_gene','_type']).to_csv(ROOT/'data/combined_mu_ms_cell_counts.csv',index=False)
    section='''<!-- combined-mu-ms:start -->
<section id="combined-mu-ms"><h2>Mu–Ms 散点图：按细胞类型着色</h2>
<p>一张图三个基因面板；每个面板叠加四类细胞，共 1,720 个细胞。颜色代表细胞类型，两个文库合并显示，所有细胞统一为圆点。横轴 Mu、纵轴 Ms；各基因使用各自坐标范围。</p>
<label>表达尺度 <select id="muMsScale"><option value="log1p">log1p(Mu) / log1p(Ms)</option><option value="linear">Mu / Ms 原始 moments 尺度</option></select></label>
<a id="muMsPdf" href="figures/three_genes_Mu_Ms_celltype_log1p.pdf">当前图 PDF</a> · <a id="muMsSvg" href="figures/three_genes_Mu_Ms_celltype_log1p.svg">SVG 放大</a>
<a id="muMsImageLink" href="figures/three_genes_Mu_Ms_celltype_log1p.svg"><img id="muMsImage" src="figures/three_genes_Mu_Ms_celltype_log1p.png" alt="FOXG1、EMX1、BCL11B 的 Mu–Ms 散点图，按完整细胞类型着色"></a>
<p>图例顺序：Radial Glia → Neuroblast → Immature Neuron → Neuron。保留所有 bins 和零值。Mu/Ms 是此前重建并保存的 moments。</p>
<script>document.getElementById('muMsScale').addEventListener('change',function(){const stem='figures/three_genes_Mu_Ms_celltype_'+this.value;document.getElementById('muMsImage').src=stem+'.png';document.getElementById('muMsImageLink').href=stem+'.svg';document.getElementById('muMsPdf').href=stem+'.pdf';document.getElementById('muMsSvg').href=stem+'.svg';});</script>
</section><h2>其他表达和 bin 分布</h2>
<!-- combined-mu-ms:end -->'''
    page=ROOT/'index.html';text=page.read_text()
    text=text.replace('<h1>FOXG1 / EMX1 / BCL11B：整体表达与全部 bins</h1>','<h1>Mu–Ms 散点图：三个基因、全部细胞类型</h1>')
    if '<!-- combined-mu-ms:start -->' in text:
        text=re.sub(r'<!-- combined-mu-ms:start -->.*?<!-- combined-mu-ms:end -->',lambda _:section,text,flags=re.S)
    else:
        assert '<div class="controls">' in text
        text=text.replace('<div class="controls">',section+'\n<div class="controls">',1)
    text=re.sub(r'<div class="note">.*?</div>','<div class="note"><strong>首先展示三个基因的 Mu–Ms 合并散点图。</strong>各细胞类型在同一基因面板内叠加，使用全称和发育顺序。可切换原始 moments 或 log1p 尺度；下方保留其他表达和 bin 分布。<br>Mu/Ms 来自已保存的重建矩阵，历史 bins 核验一致；不是确认恢复的历史连续浮点值。</div>',text,count=1,flags=re.S)
    page.write_text(text)
    report=ROOT/'REPORT_CN.md';r=report.read_text()
    addition='''<!-- combined-mu-ms:start -->
## 首图：三个基因的 Mu–Ms 散点图

[PNG](figures/three_genes_Mu_Ms_celltype_linear.png) · [PDF](figures/three_genes_Mu_Ms_celltype_linear.pdf) · [SVG](figures/three_genes_Mu_Ms_celltype_linear.svg) · [log1p 版本](figures/three_genes_Mu_Ms_celltype_log1p.png)

一张图包含 FOXG1、EMX1、BCL11B 三个面板，每个面板叠加全部四种细胞类型。横轴 Mu、纵轴 Ms；颜色表示细胞类型，两个文库合并显示，所有细胞统一为圆点；每个基因显示同一批 1,720 个细胞，包括零值和所有 bins。完整细胞类型图例按 Radial Glia → Neuroblast → Immature Neuron → Neuron 排列。各基因独立坐标范围；原始 moments 不是原始 counts，仍是此前保存的重建结果。

[绘图脚本](code/plot_combined_mu_ms.py) · [细胞数](data/combined_mu_ms_cell_counts.csv) · [核验](audit/combined_mu_ms_validation.json)。这张首图单独导出，原有 12 页 PDF 保留原有内容。
<!-- combined-mu-ms:end -->'''
    if '<!-- combined-mu-ms:start -->' in r:r=re.sub(r'<!-- combined-mu-ms:start -->.*?<!-- combined-mu-ms:end -->',lambda _:addition,r,flags=re.S)
    else:r=r.replace('\n## 三组图分别回答什么','\n'+addition+'\n\n## 三组图分别回答什么',1)
    report.write_text(r)
    audit={'status':'passed','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'genes':GENES,'cell_type_order':TYPES,'cells_per_gene':1720,'total_plotted_observations_each_figure':5160,'axis_x':'Mu','axis_y':'Ms','default_scale':'log1p moments','alternate_scale':'linear moments','color':'cell_type','marker':'uniform circle; libraries pooled','all_bins_and_zero_moments_retained':True,'outputs':outputs,'moments_recomputed':False}
    (ROOT/'audit/combined_mu_ms_validation.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    from plot_combined_mu_ms_bins import build as build_bin_colors
    build_bin_colors()
    from plot_mu_ms_case_highlights import build as build_case_highlights
    build_case_highlights()

if __name__=='__main__':build()

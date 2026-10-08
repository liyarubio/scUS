from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(name,smoke=False):
 out=folder(name,smoke);status(out,'plot','running');dest=out/'figures';dest.mkdir(exist_ok=True);metrics=pd.read_csv(out/'metrics.csv');catalog=pd.read_csv(ROOT/'configs/representations.csv');df=catalog.merge(metrics,on='representation',validate='one_to_one');meta=pd.read_csv(out/'data/cells.csv',keep_default_na=False);labels=sorted(meta.cell_type.unique());color={k:plt.get_cmap(['tab20','tab20b','tab20c'][(i//20)%3])(i%20) for i,k in enumerate(labels)}
 fig,axes=plt.subplots(1,4,figsize=(19,12),sharey=True,layout='constrained');y=np.arange(len(df));family={'existing':'#4276ae','model':'#8c63aa','expression':'#489475','velocity':'#c08b46'}
 for ax,key in zip(axes,['ARI','AMI','NMI','ASW']):
  values=df[key].to_numpy();ax.barh(y,np.nan_to_num(values,nan=0),color=[family[v] for v in df.family]);ax.set_title(key);ax.axvline(0,color='0.4',lw=.6);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
  for j,v in enumerate(values):ax.text(v if np.isfinite(v) else 0,j,' '+(f'{v:.3f}' if np.isfinite(v) else 'NA'),va='center',fontsize=8)
  lo=min(-.05,float(np.nanmin(values))-.05);hi=max(.15,float(np.nanmax(values))+.15);ax.set_xlim(lo,hi)
 axes[0].set_yticks(y,df.display,fontsize=9);axes[0].invert_yaxis();fig.suptitle(f'{name} | n={len(meta):,} | Leiden: 15-NN union, resolution=1, seed=42\nASW in native representation space; {'SMOKE TEST ONLY' if smoke else 'descriptive full-cohort analysis'}',fontsize=14)
 for ext in ['png','pdf']:fig.savefig(dest/f'01_expanded_clustering_metrics.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(6,5,figsize=(23,26),layout='constrained')
 for ax,row in zip(axes.ravel(),df.itertuples(index=False)):
  xy=np.load(out/'evaluation'/row.representation/'umap.npy')
  for lab in labels:
   use=meta.cell_type.eq(lab).to_numpy();ax.scatter(xy[use,0],xy[use,1],s=2 if len(meta)>5000 else 4,c=[color[lab]],rasterized=True,lw=0)
  ax.set_title(row.display,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
 last=axes.ravel()[-1];last.axis('off');last.legend([plt.Line2D([],[],marker='o',ls='',color=color[lab]) for lab in labels],labels,loc='center',fontsize=8,ncol=1 if len(labels)<18 else 2,frameon=False)
 fig.suptitle(f'{name} | same {len(meta):,} cells and colors across all representations\nUMAP: 15 non-self neighbors, min_dist=0.5, seed=42, 500 epochs',fontsize=16)
 for ext in ['png','pdf']:fig.savefig(dest/f'02_umap_all_representations.{ext}',dpi=160,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(18,12),sharey=True,layout='constrained')
 for ax,key in zip(axes,['ARI','AMI','NMI']):
  data=[pd.read_csv(out/'evaluation'/r/'sensitivity.csv')[key].to_numpy() for r in df.representation];ax.boxplot(data,vert=False,tick_labels=df.display if ax==axes[0] else ['']*len(data),showfliers=True);ax.set_title(f'{key}: 3 resolutions × 3 seeds');ax.invert_yaxis()
 axes[0].set_yticks(np.arange(1,len(df)+1),df.display,fontsize=9)
 for ax in axes[1:]:ax.tick_params(labelleft=False)
 for ext in ['png','pdf']:fig.savefig(dest/f'03_parameter_sensitivity.{ext}',dpi=160,bbox_inches='tight')
 plt.close(fig)
 report=f'''# {name}：全量细胞的 29 项描述性聚类比较

{'**独立 smoke 测试产物，不是正式全量结果。**' if smoke else '**正式全量结果。**'} 本次包含 {len(meta):,} 个细胞，29 项表示。

- [四指标主图 PNG](figures/01_expanded_clustering_metrics.png) / [PDF](figures/01_expanded_clustering_metrics.pdf)
- [全部表示 UMAP](figures/02_umap_all_representations.png) / [PDF](figures/02_umap_all_representations.pdf)
- [参数敏感性](figures/03_parameter_sensitivity.png)
- [指标表](metrics.csv)、[分析矩阵清单](features/manifest.csv)、[逐细胞 ID 和标签](data/cells.csv)、[原始 U+S 输入 H5AD](data/input.h5ad)

ARI/AMI/NMI 来自固定 15 个非自身近邻的无权无向并集图，Leiden resolution=1、seed42。ASW 在各表示原始度量空间上用全量细胞精确计算，未在 UMAP 空间计算。分块累积避免保存全量 N×N 矩阵；缺少共同观测特征的距离不可定义，涉及这种情况的 ASW 标 NA 并记录数量。原始分布统计不代表跨生物样本泛化。

UMAP 直接使用保存的 15 个非自身精确近邻；传入算法时额外加入自身零距离，因此内部近邻数组为 16 列。全部表示使用相同细胞、颜色、min_dist=0.5、seed42、500 epochs。图上视觉分离不是独立的模型优越性证据。

[本轮 moments 协议](data/moments_protocol.json)：U/S 分别按细胞归一化至 10,000；在 U+S 的 log 表达、HVG2,000、PCA30 上构建包含自身的 30 近邻均匀平均图。该预处理在全量目标集合上拟合，不根据评价标签调参。“原始空间 ASW”指每项表示自身定义的度量空间；scUS profile 的逐特征标准化亦属于该定义。

scUS 使用 epoch11、完整未掩蔽输入；Mu/Ms 来自本轮原始 counts 的统一处理，不能与历史 S2、Joint 或旧 1,060 细胞留出结果混称同一协议。所有标准化和校准均在本数据集合拟合且不使用细胞类型标签。三模型从同源原始 U+S counts 开始，scVI 在目标数据训练 10 epochs，scGPT 采用原方法 1,200 token CLS 协议，scFoundation 采用其官方 cell pooling。velocity 是带符号的 cell×gene 拟合矩阵，不是 velocity graph 或 UMAP 箭头。

每个表示的 `evaluation/<表示>/` 保存逐细胞 ASW、近邻、UMAP、九组参数结果以及 average-linkage 辅助诊断。大数据 average-linkage 使用固定最多 5,000 个细胞，不进入全量主图。
'''
 (out/'REPORT_CN.md').write_text(report);complete(out,'plot',name,list(dest.iterdir())+[out/'REPORT_CN.md']);status(out,'plot','complete')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();plot(a.dataset,a.smoke)

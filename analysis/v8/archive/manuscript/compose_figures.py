"""Publication typography using saved values/coordinates; one PNG per figure.

No clustering, model inference or training. The six temporal maps and historical
phase-portrait scatter marks retain source pixels; all other plots use saved data.
"""
from pathlib import Path
import os,json,hashlib,shutil,textwrap
M=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(M/'build/mpl'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
D=M/'figure_data';OUT=M/'figures_composed';OUT.mkdir(exist_ok=True)
DPI=600;WIDTH=5.5;INK='#243449';GRAY='#c5cdd5';BLUE='#397caf';RED='#d56b72'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.45,'xtick.major.width':.4,'ytick.major.width':.4,'xtick.major.size':2,'ytick.major.size':2,'axes.labelpad':2,'axes.titlepad':5,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'savefig.facecolor':'white','mathtext.fontset':'dejavusans'})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[];checks=[]
def read(name):return pd.read_csv(D/name,low_memory=False)
def src(name):
 p=(D/name) if not name.startswith(('figures/','framework_source/','revision_data/')) else M/name
 return {'source_asset':str(p.relative_to(M)),'source_sha256':sha(p)}
def save(fig,name,num,letters,inputs,method='redrawn from saved values; no coordinate recomputation'):
 # A panel includes its title, tick labels and axis labels. Place the letter
 # above that complete bounding box, not inside an axes-coordinate offset.
 fig.canvas.draw();renderer=fig.canvas.get_renderer();headers=[]
 for ax,ch,size in getattr(fig,'_panel_requests',[]):
  bb=ax.get_tightbbox(renderer);gap=4*fig.dpi/72
  x,y=fig.transFigure.inverted().transform((bb.x0,bb.y1+gap))
  label=fig.text(x,y,ch,fontsize=size,fontweight='bold',va='bottom',ha='left')
  label.set_gid('panel-letter')
  headers.append((label,bb,ax))
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 # Explicitly check artist text and legends against the canvas.
 outside=[]; hidden_ticks=set()
 for axis_ax in fig.axes:
  for axis in [axis_ax.xaxis,axis_ax.yaxis]:
   low,high=sorted(axis.get_view_interval())
   for tick in axis.get_major_ticks()+axis.get_minor_ticks():
    if not low-1e-10<=tick.get_loc()<=high+1e-10:hidden_ticks.update([tick.label1,tick.label2])
 for a in fig.findobj(matplotlib.text.Text):
  if a in hidden_ticks or not a.get_visible() or not a.get_text():continue
  b=a.get_window_extent(renderer)
  if b.width>0 and b.height>0 and (b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5):outside.append({'text':a.get_text(),'bbox':list(b.bounds),'canvas':[fig.bbox.width,fig.bbox.height]})
 panel_labels=[t for t,_,_ in headers]
 collisions=[]
 for t in panel_labels:
  bb=t.get_window_extent(renderer)
  for idx,ax in enumerate(fig.axes):
   if bb.overlaps(ax.get_window_extent(renderer)):collisions.append({'letter':t.get_text(),'axis':idx})
 text_collisions=[];legend_collisions=[]
 for label in panel_labels:
  bb=label.get_window_extent(renderer)
  for t in fig.findobj(matplotlib.text.Text):
   if t is label or t in hidden_ticks or not t.get_visible() or not t.get_text():continue
   if bb.overlaps(t.get_window_extent(renderer)):text_collisions.append([label.get_text(),t.get_text()])
  for leg in fig.findobj(matplotlib.legend.Legend):
   if leg.get_visible() and bb.overlaps(leg.get_window_extent(renderer)):legend_collisions.append(label.get_text())
 header_positions=[]
 for label,content,ax in headers:
  bb=label.get_window_extent(renderer)
  header_positions.append(dict(letter=label.get_text(),left_alignment_error_pt=(bb.x0-content.x0)*72/fig.dpi,
   gap_above_complete_panel_pt=(bb.y0-content.y1)*72/fig.dpi,
   label_bbox_figure=list(bb.transformed(fig.transFigure.inverted()).bounds),
   content_bbox_figure=list(content.transformed(fig.transFigure.inverted()).bounds)))
 check={'figure':num,'text_outside_canvas':outside,'panel_letters_over_data_axes':collisions,
  'panel_letters_over_text':text_collisions,'panel_letters_over_legends':legend_collisions,
  'panel_headers':header_positions}
 if text_collisions or legend_collisions:raise ValueError(check)
 if str(num)=='2':
  labels=fig.axes[1].get_xticklabels();bbs=[t.get_window_extent(renderer) for t in labels]
  gaps=[(r.x0-l.x1)*72/fig.dpi for l,r in zip(bbs,bbs[1:])]
  check['pairing_category_gaps_pt']=gaps
  if min(gaps)<2:raise ValueError(('pairing categories too close',gaps))
 if str(num)=='3':
  titles=[ax.title for ax in fig.axes if ax.title.get_text()];check['column_titles']=[t.get_text() for t in titles];check['titles_single_line']=all('\n' not in t.get_text() for t in titles)
  title_overlaps=[]
  for t in titles:
   for other in titles+panel_labels:
    if t is not other and t.get_window_extent(renderer).overlaps(other.get_window_extent(renderer)):title_overlaps.append([t.get_text(),other.get_text()])
  check['column_title_overlaps']=title_overlaps
  if title_overlaps or not check['titles_single_line']:raise ValueError(check)
 checks.append(check)
 if collisions:raise ValueError((num,'panel letters overlap data axes',collisions))
 if outside:raise ValueError((num,outside))
 p=OUT/name;fig.savefig(p,dpi=DPI);plt.close(fig)
 records.append({'asset':str(p.relative_to(M)),'sha256':sha(p),'size_px':list(Image.open(p).size),'dpi':DPI,'figure':str(num),'panel_letters':list(letters),'sources':[src(n) for n in inputs],'method':method})
def letter(ax,c,size=9):
 fig=ax.figure
 if not hasattr(fig,'_panel_requests'):fig._panel_requests=[]
 fig._panel_requests.append((ax,c,size))
def clean(ax):
 ax.spines[['left','bottom']].set_color(GRAY);ax.grid(False)
def box(ax,groups,colors,whis=1.5,positions=None):
 z=ax.boxplot(groups,positions=positions,widths=.5,whis=whis,showfliers=False,patch_artist=True,medianprops={'color':INK,'linewidth':.7},boxprops={'edgecolor':INK,'linewidth':.45},whiskerprops={'color':INK,'linewidth':.45},capprops={'color':INK,'linewidth':.45})
 for b,c in zip(z['boxes'],colors):b.set_facecolor(c);b.set_alpha(.5)
 clean(ax)
def legend(fig,labels,colors,loc,cols=4,size=6):
 handles=[Line2D([],[],marker='o',ls='',markersize=1.8,color=c,label=textwrap.fill(l,width=22)) for l,c in zip(labels,colors)]
 return fig.legend(handles=handles,loc='center',bbox_to_anchor=loc,ncol=cols,frameon=False,fontsize=size,handletextpad=.4,handlelength=.55,columnspacing=1.05,labelspacing=.4,borderaxespad=0)
# Supplied Figure 1 already contains panel letters; preserve it byte-for-byte.
p=M/'framework_source/Figure1_supplied.png';dst=OUT/'fig01_framework.png';shutil.copy2(p,dst)
records.append({'asset':str(dst.relative_to(M)),'sha256':sha(dst),'size_px':list(Image.open(dst).size),'figure':'1','panel_letters':list('ABCD'),'added_letters':False,'sources':[src('framework_source/Figure1_supplied.png')],'method':'author-supplied PNG, copied byte-for-byte; no additional lettering'})

# Figure 2:  same rolling statistics and exact distributions as the PPT.
f=plt.figure(figsize=(WIDTH,3.5));a=f.add_axes([.115,.605,.345,.27]);b=f.add_axes([.585,.605,.395,.27]);c=f.add_axes([.115,.13,.855,.26])
d=read('loss.csv.gz')
for key,color,label in [('loss_u',BLUE,'Unspliced'),('loss_s',RED,'Spliced')]:
 x=d[d.metric.eq(key)].sort_values('global_step');a.fill_between(x.global_step.to_numpy()/1000,x.rolling_q25.to_numpy(),x.rolling_q75.to_numpy(),color=color,alpha=.09,lw=0);a.plot(x.global_step/1000,x.rolling_median,color=color,lw=.28,alpha=.9,label=label)
a.set(xlabel='Training step (×10³)',ylabel='Masked-bin loss',ylim=(1.65,2.55));a.legend(frameon=False,loc='upper right',handlelength=1.2,fontsize=6);clean(a)
d=read('pairing.csv.gz');keys=['correct','matched_cell','cross_celltype','within_cell_gene'];box(b,[d[d.condition.eq(k)].mean_distance for k in keys],['#829abe','#e0bc76','#91b6a5','#d88893'],whis=[10,90]);b.set(xticks=range(1,5),xticklabels=['Correct\npair','Same gene\nsame type','Same gene\nother type','Different\ngene'],ylabel='Mean U–S distance',ylim=(.785,.881));b.tick_params(axis='x',labelsize=6)
d=read('epoch.csv.gz');assert len(d)==15000;box(c,[d[d.epoch.eq(k)].mean_distance for k in range(15)],[BLUE]*15,whis=[10,90]);c.set(xticks=range(1,16),xticklabels=range(15),xlabel='Training epoch',ylabel='Mean gene-wise\nU–S distance',ylim=(.06,.34))
for ax,ch in zip([a,b,c],'ABC'):letter(ax,ch)
save(f,'fig02_training.png',2,'ABC',['loss.csv.gz','pairing.csv.gz','epoch.csv.gz'])
# Figure 3: retain all saved UMAP observations and original palettes.
colors=json.loads((D/'benchmark_colors.json').read_text());reps=['raw_distance_profile','concatenated_u_s','scGPT','scFoundation'];titles=['scUS distance','scUS cell embedding','scGPT','scFoundation'];datasets=['forebrain','oscc','mouse_gastrulation'];inputs=['benchmark_colors.json'];f=plt.figure(figsize=(WIDTH,5.5))
rows=[(.76,.17,.715),(.51,.155,.463),(.275,.14,.12)]
for i,(ds,(y,h,ly)) in enumerate(zip(datasets,rows)):
 for j,rep in enumerate(reps):
  name=f'{ds}_{rep}.csv.gz';inputs.append(name);d=read(name);xy=d[['UMAP1','UMAP2']].to_numpy();order=np.random.default_rng(42).permutation(len(d));ax=f.add_axes([.09+j*.23,y,.205,h]);rgba=np.array([matplotlib.colors.to_rgba(colors[ds][v]) for v in d.label]);ax.scatter(xy[order,0],xy[order,1],c=rgba[order],s=.55 if len(d)<2000 else .06,alpha=.85,lw=0);ax.set(xticks=[],yticks=[],xlabel='UMAP 1');ax.set_ylabel('UMAP 2' if j==0 else '',fontsize=6);ax.xaxis.label.set_size(6);ax.set_title(titles[j] if i==0 else '',fontsize=6,pad=5);ax.set_aspect('equal',adjustable='datalim');ax.margins(.04);clean(ax);letter(ax,chr(65+i*4+j),size=8);checks.append({'figure':'3','dataset':ds,'representation':rep,'n_points':len(d),'saved_coordinates_used':True})
 f.text(.016,y+h/2,['Forebrain','OSCC','Mouse'][i],rotation=90,va='center',ha='center',fontsize=6,fontweight='bold')
 labels=list(colors[ds]);legend(f,labels,[colors[ds][x] for x in labels],(.53,ly),cols=4,size=5.5)
save(f,'fig03_umap.png',3,'ABCDEFGHIJKL',inputs)
# Figure 4: original matched-bin examples plus all eligible five-model contrasts.
f=plt.figure(figsize=(WIDTH,5.6));d=read('context.csv.gz');selected=read('context_selected.csv.gz');cts=['Immature Neuron','Neuroblast','Neuron','Radial Glia'];palette=colors['forebrain']
for j,g in enumerate(['EMX1','BCL11B']):
 x=d[d.gene.eq(g)];sel=selected[selected.gene.eq(g)];a=f.add_axes([.13+j*.48,.765,.335,.16]);b=f.add_axes([.13+j*.48,.465,.335,.15])
 for ct in cts:
  z=x[x.cell_type.eq(ct)];a.scatter(z.log1p_Mu,z.log1p_Ms,s=3.2,color=palette[ct],alpha=.24,lw=0)
 a.scatter(sel.log1p_Mu,sel.log1p_Ms,s=4,c=[palette[ct] for ct in sel.cell_type],lw=0);a.scatter(sel.log1p_Mu,sel.log1p_Ms,s=10,facecolors='none',edgecolors=INK,lw=.4);a.set(xlabel='log(1 + Mu)',ylabel='log(1 + Ms)',title=g);clean(a)
 groups=['Radial Glia','Neuron'] if j==0 else ['Immature Neuron','Neuron'];box(b,[sel[sel.cell_type.eq(ct)].us_cosine_distance for ct in groups],[palette[ct] for ct in groups]);rng=np.random.default_rng(42)
 for k,ct in enumerate(groups,1):
  values=sel[sel.cell_type.eq(ct)].us_cosine_distance.to_numpy();b.scatter(rng.normal(k,.065,len(values)),values,s=2.2,color=palette[ct],alpha=.45,lw=0)
 b.set(xticks=[1,2],xticklabels=['Radial glia','Neuron'] if j==0 else ['Immature neuron','Neuron'],ylabel='U–S distance');letter(a,chr(65+j));letter(b,chr(67+j));checks.append({'figure':'4','gene':g,'n_all':len(x),'n_matched':len(sel)})
legend(f,cts,[palette[x] for x in cts],(.53,.688),cols=4)
c=pd.read_csv(M/'revision_data/context_effects.csv');summ=pd.read_csv(M/'revision_data/context_model_summary.csv');models=['input_embedding','random_42','random_43','random_44','epoch_11'];labels=['Input','R42','R43','R44','Pretr.'];mc=['#b8bec5','#cfb088','#a8b9ad','#b3acc8',BLUE]
e=f.add_axes([.13,.105,.335,.22]);groups=[c[c.model.eq(m)].min_absolute_effect.to_numpy() for m in models];assert all(len(x)==64 for x in groups)
box(e,groups,mc);rng=np.random.default_rng(42)
for k,values in enumerate(groups,1):e.scatter(rng.normal(k,.065,len(values)),values,s=2.2,alpha=.55,lw=0,color=mc[k-1])
e.set(xticks=range(1,6),xticklabels=labels,ylim=(-.3,7),yticks=[0,2,4,6],ylabel="Smaller absolute effect\n(Cohen's d)",title='Matched-token effects');letter(e,'E')
ff=f.add_axes([.65,.105,.30,.22])
for k,m in enumerate(models):
 x=summ[summ.model.eq(m)].iloc[0]
 if m=='input_embedding':ff.text(52,k,'N/A',ha='center',va='center',fontsize=6,color='#7b858f')
 else:
  value=100*x.direction_fraction;ff.scatter(value,k,s=13,color=mc[k],zorder=3);ff.text(value-5,k,f'{int(x.direction_count)}/64',ha='right',va='center',fontsize=6)
ff.set(ylim=(4.6,-.6),yticks=range(5),yticklabels=labels,xlim=(0,105),xticks=[0,50,100],xlabel='Direction agreement (%)',title='Across-library directions');ff.tick_params(axis='y',length=0);clean(ff);letter(ff,'F')
checks.append(dict(figure='4',systematic_models=5,contrasts_per_model=64,genes=9,all_contrasts_retained=True,input_direction_undefined=True))
save(f,'fig04_context.png',4,'ABCDEF',['context.csv.gz','context_selected.csv.gz','benchmark_colors.json','revision_data/context_effects.csv','revision_data/context_model_summary.csv'],'original examples and all 320 model/contrast effects; direction is reported separately, no significance tests')
# Figure 5: all four panels have baseline rank tests; correction families are explicit.
f=plt.figure(figsize=(WIDTH,4.3));d=pd.read_csv(M/'revision_data/skap2_timecourse.csv');rpl=pd.read_csv(M/'revision_data/rpl23_timecourse.csv');sig=pd.read_csv(M/'revision_data/figure5_significance.csv');times=[0,75,150,300];tc=['#929daa',BLUE,'#4c9b84','#d39c55']
for row,data in enumerate([d,rpl]):
 for j in range(2):
  ch='ABCD'[2*row+j]
  column=('log1p_Mu_plus_Ms' if row==0 else 'log1p_raw_total_CPM') if j==0 else 'distance'
  title=['RNA expression','SKAP2: paired distance','RPL23: RNA expression','RPL23: paired distance'][2*row+j]
  ylabel=('log(1 + Mu + Ms)' if row==0 else 'log(1 + U+S CPM)') if j==0 else 'U–S distance'
  ax=f.add_axes([.13+j*.50,.62-row*.48,.335,.245])
  groups=[data[data.minutes.eq(t)][column].dropna().to_numpy() for t in times]
  box(ax,groups,tc);rng=np.random.default_rng(42)
  for k,values in enumerate(groups,1):ax.scatter(rng.normal(k,.06,len(values)),values,s=2.2,alpha=.24,lw=0,color=tc[k-1])
  annotations=[]
  if ch in 'ABCD':
   lo=min(v.min() for v in groups);hi=max(v.max() for v in groups);span=hi-lo
   # Three nested baseline comparisons above every observation, inside the axes.
   for level,t in enumerate(times[1:]):
    result=sig[sig.panel.eq(ch)&sig.time_min.eq(t)].iloc[0]
    right=level+2;y=hi+(.10+.16*level)*span;h=.035*span
    ax.plot([1,1,right,right],[y,y+h,y+h,y],color=INK,lw=.5,clip_on=False)
    ax.text((1+right)/2,y+h+.018*span,result.annotation,ha='center',va='bottom',fontsize=6)
    annotations.append(dict(time_min=t,q=float(result.q_cell_level),text=result.annotation))
   ax.set_ylim(lo-.08*span,hi+.63*span)
  ax.set(xticks=range(1,5),xticklabels=times,xlabel='Sampling time (min)',ylabel=ylabel,title=title);letter(ax,ch)
  checks.append(dict(figure='5',panel=ch,title=title,kind='boxplot_with_all_observed_cell_points',readout=column,counts=[len(v) for v in groups],significance=annotations))
checks.append(dict(figure='5',corrected_cells=823,group_sizes=[186,145,368,124],excluded_unstimulated_150=16,example_gene='RPL23',all_candidate_genes_reported=['RPL23','RPS8'],significance='Two-sided Mann-Whitney U; A/B retain historical BH-12; C/D use a six-test RPL23 display family',biological_replicates_verified=False))
save(f,'fig05_perturbation.png',5,'ABCD',['revision_data/skap2_timecourse.csv','revision_data/rpl23_timecourse.csv','revision_data/figure5_significance.csv'],'four cell-level box-and-scatter distributions; 1.5 IQR whiskers; A/B historical BH-12, C/D RPL23 BH-6 annotations; missing RPL23 distances excluded; no model calls')
# Figure 6:  saved UMAP coordinates, source heatmap interiors and one colorbar.
f=plt.figure(figsize=(WIDTH,5.0));pair_titles=['U–S','U–ATAC','S–ATAC'];palette={'Upper Layer':'#1f77b4','Deeper Layer':'#ff7f0e','V-SVZ':'#2ca02c','RG, Astro, OPC':'#d62728','Ependymal cells':'#9467bd','IPC':'#8c564b','Subplate':'#e377c2'};inputs=['figures/temporal_heatmaps.png']
for j,pair in enumerate(['us','ua','sa']):
 name=f'multiomics_{pair}.csv.gz';inputs.append(name);d=read(name);ax=f.add_axes([.09+j*.285,.75,.245,.17]);xy=d[['umap_x','umap_y']].to_numpy()
 for ct,color in palette.items():
  z=d[d.cell_type.eq(ct)];ax.scatter(z.umap_x,z.umap_y,s=.5,c=color,alpha=.85,lw=0)
 center=(xy.min(0)+xy.max(0))/2;span=np.ptp(xy,axis=0).max()*1.12;ax.set(xlim=(center[0]-span/2,center[0]+span/2),ylim=(center[1]-span/2,center[1]+span/2),xticks=[],yticks=[],xlabel='UMAP 1',ylabel='UMAP 2' if j==0 else '',title=pair_titles[j]);ax.set_aspect('equal');clean(ax);letter(ax,chr(65+j));assert len(d)==3365
legend(f,list(palette),list(palette.values()),(.51,.663),cols=4,size=6)
im=Image.open(M/'figures/temporal_heatmaps.png').convert('RGB');# Source axes-interior crops, excluding text and borders.
# Coordinates audited from original 3545 x 1829 PNG (PPT image25).
heat_rects=[]
for i,(y0,y1) in enumerate([(130,914),(981,1765)]):
 for j,(x0,x1) in enumerate([(64,1033),(1226,2195),(2387,3356)]):
  patch=im.crop((x0,y0,x1,y1));ax=f.add_axes([.09+j*.285,.365-i*.28,.245,.185]);ax.imshow(patch,aspect='auto',interpolation='nearest');ax.set(xticks=[],yticks=[],title=pair_titles[j]);ax.set_ylabel('Genes' if j==0 else '');ax.set_xlabel('Cells by latent time' if i==1 else '');ax.spines[['left','bottom','top','right']].set_visible(False);letter(ax,chr(68+i*3+j));heat_rects.append([x0,y0,x1,y1])
cax=f.add_axes([.925,.085,.011,.465]);cb=f.colorbar(ScalarMappable(norm=Normalize(-2.5,2.5),cmap='RdBu'),cax=cax,ticks=[-2,0,2]);cb.set_label('Row z-score',fontsize=7,labelpad=2);cb.outline.set_linewidth(.4)
save(f,'fig06_multiomics.png','6','ABCDEFGHI',inputs,'saved UMAP coordinates; six source heatmap interiors retain original pixels, order and color mapping')
records[-1]['heatmap_crop_boxes_px']=heat_rects
records.sort(key=lambda x:int(x['figure']))
(OUT/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
(OUT/'typography_audit.json').write_text(json.dumps({'panel_header_rule':'left aligned with complete panel (axes, ticks, labels and title); separate header 4 pt above content','minimum_panel_header_gap_pt':4,'font_family':'DejaVu Sans','axis_labels_pt':7,'tick_and_legend_pt':6,'panel_titles_pt':8,'panel_letters_pt':9,'legend_marker_diameter_pt':1.8,'figure3_dense_mouse_legend_pt':5.5,'figure3_titles_pt':6,'figure3_axis_labels_pt':6,'figure2A_linewidth_pt':.28,'figure4_all_points_area_pt2':3.2,'figure4_matched_points_area_pt2':4,'figure4_highlight_ring_area_pt2':10,'figure4_lower_points_area_pt2':2.2,'exceptions':['Figure 1: supplied raster used verbatim, including its embedded fonts','Figure 3: 6 pt one-line column titles and axes, 5.5 pt legend text, 8 pt panel letters'],'checks':checks},indent=2)+'\n')
print('Built six PNG figures. Saved coordinates/observations retained; legend symbols 1.8 pt.')

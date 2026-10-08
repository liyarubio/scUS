"""Cell-type Mu/Ms scatter with exact original Figure 4B cells outlined."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plot_combined_mu_ms import ROOT, GENES, TYPES, COLORS

SOURCE = Path('/data1/liyaru/scUS/outputs/paper/forebrain_gene_context_examples')

def replace_section(text, name, section, before):
    pattern = rf'<!-- {name}:start -->.*?<!-- {name}:end -->'
    if re.search(pattern, text, flags=re.S):
        return re.sub(pattern, lambda _: section, text, count=1, flags=re.S)
    assert before in text
    return text.replace(before, section + '\n' + before, 1)

def build():
    source = ROOT / 'data/all_cells_expression.csv'
    data = pd.read_csv(source)
    manifest = pd.read_csv(SOURCE / 'spotlight_manifest.csv').set_index('gene')
    original = pd.read_csv(SOURCE / 'spotlight_cell_values.csv')
    selected = data.loc[data.original_B_cell].copy()
    expected = original.loc[original.gene.isin(GENES)]
    assert set(zip(selected.gene, selected.cell_id)) == set(zip(expected.gene, expected.cell_id))
    assert data.groupby('gene').size().to_dict() == dict.fromkeys(GENES, 1720)
    assert np.isfinite(selected.us_cosine_distance).all()
    for gene in GENES:
        p = data[data.gene == gene]
        case = manifest.loc[gene]
        mask = (p.u_bin == case.u_bin) & (p.s_bin == case.s_bin) & p.cell_type.isin([case.type_a, case.type_b])
        assert np.array_equal(mask.to_numpy(), p.original_B_cell.to_numpy())
    summary = selected.groupby(['gene', 'cell_type'], sort=False).agg(
        n=('cell_id', 'size'), u_bin=('u_bin', 'first'), s_bin=('s_bin', 'first'),
        distance_mean=('us_cosine_distance', 'mean'), distance_median=('us_cosine_distance', 'median'),
        distance_q25=('us_cosine_distance', lambda x: x.quantile(.25)),
        distance_q75=('us_cosine_distance', lambda x: x.quantile(.75))).reset_index()
    selected.to_csv(ROOT / 'data/original_B_highlighted_cells.csv', index=False)
    summary.to_csv(ROOT / 'data/original_B_highlighted_distance_summary.csv', index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':'#21364A','text.color':'#21364A',
        'axes.titlecolor':'#21364A','pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    # Select additional groups using coverage only; effects never enter the ranking.
    eligible = pd.read_csv(ROOT.parent / 'figure4b_all_contrasts_20260924/data/all_eligible_contrasts.csv')
    count_cols = [c for c in eligible.columns if c.startswith(('n_a_', 'n_b_'))]
    eligible['minimum_cell_count'] = eligible[count_cols].min(axis=1)
    eligible['total_cell_count'] = eligible[count_cols].sum(axis=1)
    case_sets = [manifest.loc[GENES].copy()]
    extras = {}
    for gene in GENES:
        q = eligible[(eligible.gene_name == gene) & (~eligible.original_B)].copy()
        original_types = {manifest.loc[gene,'type_a'], manifest.loc[gene,'type_b']}
        if gene != 'BCL11B':
            q = q[[{a,b} == original_types for a,b in zip(q.type_a,q.type_b)]]
        else:
            # The original two cell types occupy only (15,15); use the two remaining eligible contrasts.
            orig_cells = data[(data.gene==gene) & data.cell_type.isin(original_types)]
            assert set(zip(orig_cells.u_bin,orig_cells.s_bin)) == {(15,15)}
        q = q[q.minimum_cell_count >= 10].sort_values(
            ['minimum_cell_count','total_cell_count','u_bin','s_bin','type_a','type_b'],
            ascending=[False,False,True,True,True,True]).head(2)
        assert len(q)==2
        extras[gene] = q
    for j in range(2):
        rows=[]
        for gene in GENES:
            row=extras[gene].iloc[j].copy();row['gene']=gene;rows.append(row)
        case_sets.append(pd.DataFrame(rows).set_index('gene'))
    case_rows=[];case_cells=[]
    for group_id,m in enumerate(case_sets):
        for gene in GENES:
            case=m.loc[gene]
            c=data[(data.gene==gene)&(data.u_bin==case.u_bin)&(data.s_bin==case.s_bin)&data.cell_type.isin([case.type_a,case.type_b])].copy()
            coverage=c.groupby(['sample','cell_type']).size()
            assert len(coverage)==4 and coverage.min()>=10
            assert np.isfinite(c.us_cosine_distance).all()
            c['case_group']=group_id;case_cells.append(c)
            case_rows.append(dict(case_group=group_id,gene=gene,u_bin=int(case.u_bin),s_bin=int(case.s_bin),
                type_a=case.type_a,type_b=case.type_b,n=int(len(c)),minimum_per_library_type=int(coverage.min()),
                selection='original' if group_id==0 else 'coverage ranking; distance not used'))
    case_table=pd.DataFrame(case_rows)
    case_table.to_csv(ROOT/'data/three_case_groups_manifest.csv',index=False)
    pd.concat(case_cells,ignore_index=True).to_csv(ROOT/'data/three_case_groups_cells.csv',index=False)
    case_summary=pd.concat(case_cells).groupby(['case_group','gene','cell_type']).us_cosine_distance.agg(['size','mean','median']).reset_index()
    case_summary.to_csv(ROOT/'data/three_case_groups_distance_summary.csv',index=False)
    outputs = []
    books = {scale: PdfPages(ROOT / f'figures/three_genes_Mu_Ms_all_cases_{scale}.pdf') for scale in ['log1p','linear']}
    for group_id, case_manifest in enumerate(case_sets):
      for scale in ['log1p', 'linear']:
        fig = plt.figure(figsize=(18, 10.8))
        grid = fig.add_gridspec(2, 3, height_ratios=[2.25, 1], left=.065, right=.98,
                               bottom=.17, top=.77, hspace=.48, wspace=.29)
        fig.suptitle('Mu-Ms expression by cell type: ' + ('original Figure 4B cases' if group_id == 0 else f'additional cases {group_id}'), fontsize=21, y=.98)
        handles = [Line2D([], [], marker='o', ls='', color=COLORS[t], markersize=8, label=t) for t in TYPES]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.945),
                   ncol=4, frameon=False, fontsize=12)
        fig.text(.5,.883,'Black rings = cells in the selected bin and cell-type contrast; all other cells remain visible.',
                 ha='center', fontsize=12)
        fig.text(.5,.853,'Libraries pooled; cell-type colors and developmental display order are shared across panels.',
                 ha='center', fontsize=11, color='#526478')
        for col, gene in enumerate(GENES):
            ax = fig.add_subplot(grid[0,col]); bx = fig.add_subplot(grid[1,col])
            p = data[data.gene == gene].sample(frac=1, random_state=20260924)
            case = case_manifest.loc[gene]
            selected_mask = (p.u_bin == case.u_bin) & (p.s_bin == case.s_bin) & p.cell_type.isin([case.type_a, case.type_b])
            c = p[selected_mask]
            types = [t for t in TYPES if t in [case.type_a, case.type_b]]
            x = p.Mu.to_numpy(); y = p.Ms.to_numpy()
            if scale == 'log1p': x, y = np.log1p(x), np.log1p(y)
            hit = selected_mask.to_numpy()
            all_points = ax.scatter(x, y, c=p.cell_type.map(COLORS), s=16, alpha=.48,
                                   edgecolors='white', linewidths=.12)
            all_points.set_gid(f'all_cells_{gene}')
            ax.scatter(x[hit], y[hit], c=c.cell_type.map(COLORS), s=22, alpha=1,
                       edgecolors='none', zorder=3)
            rings = ax.scatter(x[hit], y[hit], facecolors='none', edgecolors='#18222C',
                               s=34, linewidths=.55, zorder=4)
            rings.set_gid(f'case_{group_id}_rings_{gene}')
            assert len(all_points.get_offsets()) == 1720 and len(rings.get_offsets()) == len(c)
            ax.set(xlabel='log1p(Mu)' if scale=='log1p' else 'Mu',
                   ylabel='log1p(Ms)' if scale=='log1p' else 'Ms')
            ax.set_title(f'{gene} | U-bin = {case.u_bin}, S-bin = {case.s_bin}\n'
                         f'All n = 1,720; circled n = {len(c):,}', fontsize=14, pad=12)
            ax.grid(alpha=.15); ax.set_axisbelow(True)
            xmax=max(float(x.max()),.01); ymax=max(float(y.max()),.01)
            ax.set_xlim(-.025*xmax,1.06*xmax); ax.set_ylim(-.025*ymax,1.06*ymax)
            rng = np.random.default_rng(20260924)
            arrays = [c.loc[c.cell_type==t,'us_cosine_distance'].to_numpy() for t in types]
            for i,(t,v) in enumerate(zip(types,arrays)):
                bx.scatter(rng.normal(i,.065,len(v)), v, s=8, color=COLORS[t], alpha=.35, linewidths=0)
            boxes=bx.boxplot(arrays, positions=range(2), widths=.38, patch_artist=True,
                             showfliers=False, medianprops={'color':'#18222C','linewidth':1.6},
                             boxprops={'edgecolor':'#21364A'},whiskerprops={'color':'#21364A'},capprops={'color':'#21364A'})
            for patch,t in zip(boxes['boxes'],types): patch.set_facecolor(COLORS[t]);patch.set_alpha(.9)
            bx.set_xticks(range(2), [f'{t}\nn = {len(v):,}' for t,v in zip(types,arrays)], fontsize=11)
            bx.set_ylabel('U-S cosine distance')
            bx.set_title('Distance of the circled cells', fontsize=12, pad=9)
            bx.set_xlim(-.55,1.55);bx.grid(axis='y',alpha=.15);bx.set_axisbelow(True)
        fig.text(.065,.06,
            'Rings select the indicated bin pair AND the two cell types in each case; cells of other types with the same bins are not circled.\n'
            'Bottom: pooled descriptive distributions for the same circled cells (box = quartiles; line = median; whiskers = 1.5 IQR).\n'
            'Matching bins does not imply matching continuous expression. Mu/Ms are archived reconstructed moments; distances are from epoch 11.\n'
            'Added FOXG1/EMX1 cases use other bins; added BCL11B cases use other cell-type pairs at U=15/S=15. No new significance test.', fontsize=10, linespacing=1.55)
        fig.canvas.draw()
        books[scale].savefig(fig)
        tag = 'cases' if group_id == 0 else f'cases_added{group_id}'
        for ext in ['png','pdf','svg']:
            path=ROOT / f'figures/three_genes_Mu_Ms_celltype_{tag}_{scale}.{ext}'
            fig.savefig(path, dpi=190);outputs.append(str(path.relative_to(ROOT)))
        plt.close(fig)
    for book in books.values(): book.close()
    section='''<!-- mu-ms-cases:start -->
<section id="mu-ms-cases"><h2>细胞类型着色：原 case + 两组新增对照</h2>
<p>颜色与前面的 Forebrain UMAP 一致：Radial Glia 红、Neuroblast 橙、Immature Neuron 蓝、Neuron 绿；图例按发育顺序排列。每个基因显示全部 1,720 个细胞，黑圈标出当前对照的细胞，下排为同一批细胞的 distance 分布。两个文库合并显示。</p>
<label>选择对照 <select id="caseGroup"><option value="cases">原 Figure 4B</option><option value="cases_added1">新增第 1 组</option><option value="cases_added2">新增第 2 组</option></select></label>
<label>表达尺度 <select id="caseScale"><option value="log1p">log1p(Mu) / log1p(Ms)</option><option value="linear">Mu / Ms</option></select></label>
<a id="casePdf" href="figures/three_genes_Mu_Ms_celltype_cases_log1p.pdf">当前图 PDF</a> · <a id="caseSvg" href="figures/three_genes_Mu_Ms_celltype_cases_log1p.svg">SVG 放大</a> · <a href="figures/three_genes_Mu_Ms_all_cases_log1p.pdf">三组对照 PDF（3 页）</a> · <a href="data/three_case_groups_manifest.csv">选择清单</a> · <a href="data/three_case_groups_cells.csv">圈选细胞</a>
<a id="caseImageLink" href="figures/three_genes_Mu_Ms_celltype_cases_log1p.svg"><img id="caseImage" src="figures/three_genes_Mu_Ms_celltype_cases_log1p.png" alt="按细胞类型着色的 Mu–Ms 圈选散点图及相应 distance 分布"></a>
<p>新增组按覆盖细胞数选择：先按两个文库 × 两个细胞类型中的最小细胞数降序，再按总数降序；每个文库每种类型至少 10 个细胞，未按 distance 或显著性挑选。</p>
<p><strong>BCL11B 的原两类细胞只有 U=15/S=15，因此新增的是同一 bin 内另外两组细胞类型对照，不能解释为新增了两个不同 bin。</strong>不同对照可能重复使用同一批细胞，合并后的 distance 分布仅作描述；相同 bins 不代表连续表达等效。</p>
CASE_TABLE
<script>function updateCasePlot(){const stem='figures/three_genes_Mu_Ms_celltype_'+document.getElementById('caseGroup').value+'_'+document.getElementById('caseScale').value;document.getElementById('caseImage').src=stem+'.png';document.getElementById('caseImageLink').href=stem+'.svg';document.getElementById('casePdf').href=stem+'.pdf';document.getElementById('caseSvg').href=stem+'.svg';}document.getElementById('caseScale').addEventListener('change',updateCasePlot);document.getElementById('caseGroup').addEventListener('change',updateCasePlot);</script>
</section>
<!-- mu-ms-cases:end -->'''
    display=case_table[['case_group','gene','u_bin','s_bin','type_a','type_b','n','minimum_per_library_type']].copy()
    display['case_group']=display.case_group.map({0:'原图',1:'新增 1',2:'新增 2'})
    section=section.replace('CASE_TABLE','<div class="scroll">'+display.to_html(index=False,border=0)+'</div>')
    page=ROOT/'index.html'
    html=replace_section(page.read_text(),'mu-ms-cases',section,'<!-- combined-mu-ms:start -->')
    html=html.replace('首图按 U-bin、S-bin 两者共同确定的组着色。','首图按细胞类型着色，可切换原 case 和两组新增对照；下一张图按 U/S 联合 bin 着色。')
    page.write_text(html)
    report=ROOT/'REPORT_CN.md'
    note='''<!-- mu-ms-cases:start -->
## 细胞类型着色：原 case + 两组新增对照

[三组对照 PDF（3 页）](figures/three_genes_Mu_Ms_all_cases_log1p.pdf) · [原 case](figures/three_genes_Mu_Ms_celltype_cases_log1p.png) · [新增 1](figures/three_genes_Mu_Ms_celltype_cases_added1_log1p.png) · [新增 2](figures/three_genes_Mu_Ms_celltype_cases_added2_log1p.png)

配色与参考 Forebrain UMAP 的 `data/colors.json` 完全一致：Radial Glia #d62728、Neuroblast #ff7f0e、Immature Neuron #1f77b4、Neuron #2ca02c。图例仍按发育顺序排列。每个基因显示全部 1,720 个细胞，两个文库合并；黑圈仅标记当前 bin 和两种细胞类型对应的细胞，下排显示同一批细胞的 epoch 11 distance。

保留原 Figure 4B 的精确细胞集合，额外每基因选择两组；选择仅依赖覆盖数量，不使用 distance、效应量或显著性。候选要求两个文库中每种比较细胞类型至少 10 个细胞；排序先比较四个文库/类型组中的最小数量，再比较总数量，最后用 bin/类型名打破并列。

FOXG1/EMX1 固定原细胞类型对照，分别增加 U5/S15、U4/S15 与 U9/S14、U8/S14。BCL11B 原两类细胞只出现 U15/S15；新增全部另外两种合格类型对照（同样 U15/S15），没有伪造其他 bin。

SELECTED_TABLE

不同对照可能共享细胞，并非独立生物重复。合并的箱线图仅作描述，未新增显著性检验；相同 bins 不等于连续表达等效。Mu/Ms 仍为此前重建缓存，未重算 moments 或 bins。

[选择清单](data/three_case_groups_manifest.csv) · [圈选细胞](data/three_case_groups_cells.csv) · [distance 汇总](data/three_case_groups_distance_summary.csv) · [脚本](code/plot_mu_ms_case_highlights.py) · [核验](audit/original_B_highlights_validation.json)
<!-- mu-ms-cases:end -->'''
    table_lines = ['| ' + ' | '.join(display.columns) + ' |', '| ' + ' | '.join(['---'] * len(display.columns)) + ' |']
    table_lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in display.itertuples(index=False, name=None)]
    note=note.replace('SELECTED_TABLE', '\n'.join(table_lines))
    report.write_text(replace_section(report.read_text(),'mu-ms-cases',note,'<!-- joint-bin-color:start -->'))
    audit={'status':'passed','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'selection_matches_original_spotlight_cells':True,'selection':'original bins AND original two cell types',
        'cells_per_gene':1720,'circled_counts':selected.groupby('gene').size().to_dict(),
        'all_selected_distances_finite':True,'libraries_pooled':True,'color_order':TYPES,
        'moments_recomputed':False,'bins_recomputed':False,'outputs':outputs,
        'color_mapping':COLORS,'case_groups':case_rows,'selection_uses_distance':False,
        'minimum_cells_per_library_type':10,'bcl11b_added_groups':'other cell-type pairs at the same bin (15,15)'}
    (ROOT/'audit/original_B_highlights_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    build()

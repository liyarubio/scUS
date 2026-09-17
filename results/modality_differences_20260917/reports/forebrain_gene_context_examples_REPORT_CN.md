# Forebrain固定Token的逐基因Contextual Hidden审计

## 结果入口

- 固定token实例：U hidden（未随本结果包发布）；S hidden（未随本结果包发布）；[U-S distance](../figures/forebrain_gene_context_examples/01_fixed_token_hidden_examples_distance.png)
- [同sample全部待比较细胞的U/S bins＋固定token distance](../figures/forebrain_gene_context_examples/01B_fixed_token_distance_with_same_sample_bins.png)
- 跨sample复现（未随本结果包发布）
- [Input／random-init／epoch11归因](../figures/forebrain_gene_context_examples/03_input_random_pretrained_attribution.png)
- 共同Mu-UMAP上的输入与hidden residual（未随本结果包发布）
- [前脑panel与匹配随机panel](../figures/forebrain_gene_context_examples/05_forebrain_panel_vs_matched_null.png)
- U、S与cosine distance对照（未随本结果包发布）

## 固定口径

- 1,720个Forebrain细胞；完整原始token context；无top-k、无重新训练。
- 20个生物学定义基因全部报告；6个实例只用于提高图的可读性。
- 固定的是模型实际接收的gene ID、U/S bins和modality token，不是连续Mu/Ms或原始count逐值等同。
- hidden mean与hidden sum只相差常数128；完整128维结果用于检查固定投影是否造成误导。

## 当前汇总

- 全部可评估基因中，Epoch11 U-hidden合格对比的跨sample方向一致率：**0.903**。
- 前脑panel的64个可复现exact-bin对比中，U hidden／S hidden／U-S distance方向一致率分别为 **0.922／0.969／0.953**。
- 前脑panel median context effect：**0.6790**；位于匹配随机panel的 **26.6%** 分位。
- 前脑panel median epoch11−random增量：**-0.0108**；位于匹配随机panel的 **33.8%** 分位。
- 20个panel基因中有 **9** 个满足严格的双sample exact-bin比较条件；其余不足条件的基因保留为NA。

## 逐图解读

1. `01`：在目标gene ID和U/S bins完全相同的前提下，六个实例的epoch11 U hidden、S hidden及U-S distance仍按cell type分离，并在两个sample中方向一致。这是context dependence的直观证据。Distance图与U/S图使用完全相同的barcode集合。`01B`额外展示同sample、相同两个cell type的全部细胞bin分布；这些额外细胞不进入右侧fixed-token distance统计。
2. `02`：绝大多数panel exact-bin效应落在同号象限并靠近对角线，表明效应不是单个sample偶然产生；灰点是方向不一致的反例，继续保留。
3. `03`：Input embedding为零效应；random-init已经产生context effect，而epoch11只在部分基因／读出上更强。完整128维与hidden mean的排名并不完全相同，说明单一固定投影会改变结论。
4. `04`：在同一Mu-profile UMAP坐标上，hidden residual在扣除sample与exact-bin均值后仍呈现空间结构。该图只作描述，不参与效应检验。
5. `05`：红线没有进入匹配随机panel右尾，故不支持前脑基因集合特异增强；contextualization更像是广泛的逐基因现象。
6. `06`：U hidden、S hidden和U-S cosine distance均可对context变化作出响应，但不同基因主要由哪一侧驱动并不一致；distance不能被简化为单侧hidden变化。

## 解释边界

若epoch11不超过三个random-init，则只能说明Transformer结构和tokenization产生context sensitivity。若前脑panel不超过匹配随机panel，则contextualization是广泛现象而非前脑基因特异。任何正结果都不等价于表达独立性、因果调控或RNA动力学方向。

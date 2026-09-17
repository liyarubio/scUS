# scUS关键结果：模态间的差异本身也能表示细胞

整理日期：2026-09-17。本文围绕最新讨论重新组织已有实测结果，不新增实验、不重新选择checkpoint，也不覆盖历史摘要。论文题目暂不确定。

## 1. 核心主题

> **两个模态之间的表征差异，本身也可以携带细胞状态信息，构成一种细胞表示。**

英文工作表述：**Differences between paired RNA representations can themselves encode cell state.**

scUS不只使用U或S各自的embedding，也不只对两者拼接或平均，而是把同一细胞、同一基因的U/S embedding之间的距离保留下来，形成一个逐基因的细胞profile。

重点不是“distance越大越好”，也不是“高distance意味着异常”，而是：**哪些基因的U/S表征接近、哪些不同，这个差异模式能否描述细胞状态？**

据此，文章主线调整为：

1. 构造差异profile，并验证它能表示细胞。
2. 固定目标token，检验差异如何受到细胞背景调制。
3. 检验这种表示能否关联真实发育阶段。
4. 研究微调和模态对齐如何改变这种差异及其用途。

模型审计是解释结果的手段，Align是差异如何被改变的实验，不再取代“差异可以成为表示”这一中心。

## 2. 研究对象与术语

对于细胞c和基因g，完整细胞输入共同编码后得到两个128维向量：

\[
h_U(c,g),h_S(c,g)\in\mathbb R^{128},\qquad
d(c,g)=1-\cos\big(h_U(c,g),h_S(c,g)\big).
\]

细胞表示为：

\[
\mathbf d(c)=\big[d(c,g_1),\ldots,d(c,g_G)\big].
\]

| 对象 | 维度/操作 | 这里回答的问题 |
|---|---|---|
| U/S token hidden | cell × gene × 128，各一个模态 | 每个基因在完整输入背景中的表征是什么？ |
| Pooled U/S cell embedding | 沿基因汇总，通常cell × 128 | 单个模态的整体细胞表征是什么？ |
| U/S distance profile | 两个128维向量逐pair计算cosine distance，cell × gene | 每个细胞的逐基因跨模态差异模式是什么？ |
| Distance scalar | 本文Forebrain分类协议取每细胞有效基因distance中位数，cell × 1 | 这个细胞整体距离有多大？ |
| Hidden mean/sum profile | 沿128维求均值/和，cell × gene | 固定坐标投影下，每个gene hidden的汇总是多少？ |

Distance profile不等于U−S表达差值，也不等于沿hidden维度相加。其“gene-resolved”特征来自保留gene轴；原始表达同样保留gene轴，故不能仅以此声称创新。

本轮主对象是冻结预训练epoch11的hidden h。Adapter-only和联合微调结果另用其最终输出z计算距离，不混称zero-shot。不存在有效pair的位置保持NaN，不当作真实零距离。

## 3. 关键结果一：差异profile具有细胞状态表征能力

### 3.1 固定协议与数值

Forebrain共1,720细胞；参考sample `10X_17_029` 的660细胞，预测测试sample `10X_17_028` 的1,060细胞。主指标为4类cell type的balanced accuracy（BA）。使用距离加权15-NN；profile在参考sample内拟合标准化，以共同有效维度RMS比较细胞；pair mask使用Jaccard。分类不在UMAP坐标上进行。

| 表示 | BA | 如何解读 |
|---|---:|---|
| **Raw distance profile** | **0.8682** | 逐基因的跨模态差异可用于细胞分类 |
| **Calibrated distance profile** | **0.8925** | 训练参考内的gene/bin基线校准改善了这一读出 |
| Distance scalar | 0.3492 | 压成单一总体距离后，分类信息大幅减少 |
| Gene-shuffled distance profile | 0.2464 | 逐细胞打乱gene位置后，本协议分类接近4类BA机会水平 |
| Pooled state embedding | 0.8604 | Raw profile略高于这一pooled读出，但未据此做显著性结论 |
| Pooled U / S / concat | 0.7673 / 0.7649 / 0.7828 | Profile在本协议高于这些固定pooling结果 |
| Gene-level hidden-sum U / S / concat | 0.8834 / 0.8908 / 0.8878 | 保留gene轴的其他hidden读出也有较强表现 |
| 直接U-bin−S-bin | 0.9052 | 强输入基线仍优于distance |
| Pair-observation mask | 0.9018 | 检测/覆盖模式本身携带明显细胞状态信息 |
| U-bin / S-bin profile | 0.9033 / 0.9060 | 单模态输入也能较好分类 |
| Mu / Ms profile | 0.9080 / 0.9029 | 连续moments仍是强基线 |

Raw、calibrated和direct-bin比较使用10,011个合格pair基因；单模态U/Mu、S/Ms分别保留10,703、12,290个有效基因，覆盖不完全相同。Pooled embedding复用原10,692基因面板内有效pair的汇总。因此它们是实际读出方案比较，不是控制了所有维数、覆盖与聚合因素的单变量实验。

Calibrated distance仅在参考sample估计gene median、U/S-bin组合残差与gene MAD，是条件残差而不是新的模型或异常概率。

### 3.2 这条结果的亮点

> **不把两个模态合成一个平均向量，而保留两者的逐基因差异，同样可以获得有用的细胞表示。**

这里有信息的是差异在基因之间的分布模式，不是一个普适的“整体不一致程度”。这是本文最直接的应用证据。

必须同时说明：

- Profile相对scalar也保留了更多维度和信息，优势不单独证明cosine公式不可替代。
- 当前gene-shuffle实现逐细胞排列整行，NaN位置也随之改变；该对照同时破坏gene对应与缺失模式对应，不能单独归因为gene identity的纯效应。参见[实现](source_snapshot/iclr_distance.py)。
- 输入mask较强，说明必须认真控制覆盖模式；不能将profile分类全部解释成新的生物关系。
- Mu/Ms来自历史流程重建的全细胞无标签联合平滑；参考内拟合分类预处理并不使整个输入处理成为严格inductive流程。预训练语料暴露也未知。

**结果路径：** [扩充A图](figures/relational_geometry_story/01A_forebrain_cell_state_expanded.png) · [17种多维表示UMAP](figures/cell_state_hidden_sum/02_all_representations_umap_celltype.png) · [BA/F1/AUPRC源表](tables/cell_state_hidden_sum/profile_metrics_combined.csv) · [固定分类协议报告](reports/forebrain_REPORT_CN.md)。UMAP仅辅助展示，不作为上述分类成绩或生物机制的依据。

## 4. 关键结果二：相同目标tokens可以产生不同的distance

### 4.1 实验排除了什么？

使用全部1,720个Forebrain细胞的完整原始token context，不截断背景基因。比较同sample内同gene ID、相同U-bin/S-bin、相同modality tokens，但cell type不同的细胞。

固定的是目标基因的模型输入，而不是整个细胞输入；相同bins也不等于连续Mu/Ms或原始counts逐值相同。这是固定目标token的观察性比较，不是随机化的生物干预。

| 展示基因 | 固定U/S bins | 两个sample中的distance方向 |
|---|---|---|
| FOXG1 | 6 / 15 | Radial Glia > Immature Neuron |
| PAX6 | 11 / 15 | Radial Glia > Neuroblast |
| EMX1 | 10 / 14 | Neuron > Radial Glia |
| LHX2 | 13 / 15 | Radial Glia > Neuroblast |
| DCX | 11 / 15 | Neuron > Immature Neuron |
| BCL11B | 15 / 15 | Neuron > Immature Neuron |

20个预定义前脑基因全部保留报告，其中9个满足严格双sample的exact-bin比较条件，共64个合格gene × bin × cell-type contrast。U/S distance效应跨sample方向一致率为 **0.953125，即61/64**；U hidden、S hidden对应一致率为0.921875和0.96875。多个对比共享基因和细胞，不将64个对比当作64个独立生物重复。

### 4.2 这条结果的亮点

> **目标U/S数值tokens相同，并不决定最终distance相同；完整细胞背景可以改变两个目标token之间的表征差异。**

它解释了distance为何可能关联细胞状态：它不是只对该基因两个输入bin做局部计算，而是对已经经过全细胞context编码的两个向量进行比较。

不同基因的组间方向不必相同。上述实例也不支持一个统一的“distance越高越成熟”或“越高越不稳定”标尺。

### 4.3 预训练贡献不能跳过随机对照

- 固定token的input embedding效应为0；随机Transformer也能产生context effect。
- 前脑panel的median context effect为0.6790，位于匹配随机gene panels的26.6%分位，不支持前脑panel特异增强。
- Panel median epoch11−random增量为−0.0108，位于匹配随机panels的33.8%分位；这是报告所定义的context汇总指标，不应改写为distance分类性能增量。
- 因此，当前可以说“背景依赖的差异表示”；不能说已经证明“预训练特异地学习了前脑调控关系”。

**结果路径：** [固定token distance主图](figures/forebrain_gene_context_examples/01_fixed_token_hidden_examples_distance.png) · [同sample其他细胞bins＋固定token distance](figures/forebrain_gene_context_examples/01B_fixed_token_distance_with_same_sample_bins.png) · [Input/random/epoch11归因](figures/forebrain_gene_context_examples/03_input_random_pretrained_attribution.png) · [匹配随机panel](figures/forebrain_gene_context_examples/05_forebrain_panel_vs_matched_null.png)。

**统计与方法：** [逐对比源表](tables/forebrain_gene_context_examples/replicated_scalar_effects.csv) · [汇总JSON](tables/forebrain_gene_context_examples/analysis_summary.json) · [完整报告](reports/forebrain_gene_context_examples_REPORT_CN.md)。六个实例是展示，不替代全部panel和不满足样本量的记录。

## 5. 支持性结果：差异profile能组织阶段相关的基因模块

Erythroid的27个sample采用leave-one-sample-out。每fold仅在26个训练sample拟合标准化、K=20基因模块和Ridge阶段预测模型，评价被留出的sample。

| 模块输入 | 27个sample的阶段预测Spearman |
|---|---:|
| **Raw distance** | **0.8322** |
| S-bin | 0.8714 |
| U/S-bin concat | 0.8693 |
| 绝对bin差 | 0.8486 |
| Pair mask | 0.8405 |
| 有符号bin差 | 0.8411 |

这支持“差异表示可以关联真实发育阶段，并以基因模块方式汇总”。但distance没有超过同协议强输入基线；它验证的是红系数据上的模块分析流程，不是同一组前脑模块基因的跨组织复现。

0.8322属于**模块＋Ridge**，不等于profile 15-NN结果。历史moments处理和预训练暴露限制仍然存在，不能称为严格未见数据泛化，也不能把阶段相关解释成真实velocity或方向。

**结果路径：** [模块与阶段图](figures/erythroid/01_module_stage_validation.png) · [同协议源表](tables/erythroid/module_stage_metrics.csv) · [方法与限制](reports/erythroid_README_CN.md)。

## 6. 微调结果：差异可以被压缩，但“更接近”不是充分的成功标准

### 6.1 四种策略改变差异的方式不同

以下多层级分析固定1,720细胞、10,692基因，四模型缺失mask相同，共10,865,467个有效pair。

| 策略 | Cell-level distance中位数 | Pooled U/S 30-NN overlap |
|---|---:|---:|
| Zero-shot | 0.8566 | 0.5705 |
| Adapter-only | 0.0036 | 0.4945 |
| Joint epoch29 | 0.2326 | 0.5282 |
| Reconstruction-only epoch29 | 0.8720 | 0.5893 |

第一列先在每个细胞内取有效gene distance中位数，再对细胞取中位数。第二列是在pooled U、S cell embedding分别逐维标准化后，以RMS构建30-NN并比较邻居集合；它不是distance profile分类成绩，也不是相对Zero-shot的邻域保持率。

Adapter-only的逐pair角度接近0，并没有伴随更高的pooled U/S邻域一致性。由此可以说：**局部pair角度收缩与细胞间结构一致性不是同一个目标。** 不能仅凭此断言差异越大越好，或证明模态对齐必然删除生物信息。

### 6.2 联合微调显示任务收益与几何改变可以同时发生

| 指标 | Zero-shot | Joint epoch29 | Joint epoch49 |
|---|---:|---:|---:|
| 外层留出sample分类BA | 0.8682 | **0.9082** | 0.8949 |
| 内部固定mask验证重建CE | 2.0952 | 2.0453 | **2.0304** |
| Correct-vs-shuffle标准化效应 | 6.205 | 2.277 | 2.090 |
| 相对Zero-shot的pooled state邻域保持率 | 1.000 | 0.662 | 0.607 |

同一历史轨迹中，epoch29的分类BA比Zero-shot高约4.00个百分点；继续到epoch49，重建CE继续改善，但BA下降约1.33个百分点。不同指标不是同一单位，外层分类与内部验证QC也不是同一评价集合。

适合本文主题的解释是：**差异表示的尺度、结构和任务用途可以随适配以不同方式变化，不能用单一loss或平均distance判断其价值。**

这些是历史seed42方案级证据；冻结范围、历史批次日程和训练暴露并不完全相同，严格同预算多目标、多seed比较尚不能由此代替。Zero-shot几何不是生物真值，其变化不自动等于破坏。这里也不按外层测试结果重新选择checkpoint。

**结果路径：** [四策略多层级总览](figures/us_distance_multilevel_finetune/00_multilevel_distance_overview.png) · [cell/gene/pair源表](tables/us_distance_multilevel_finetune/level_distribution_summary.csv) · [pooled U/S邻域源表](tables/four_model_us_neighbors/consistency_summary.csv) · [任务—几何图](figures/relational_geometry_story/04_task_geometry_tradeoff.png) · [同轨迹源表](tables/relational_geometry_story/04_task_geometry_tradeoff_data.csv)。

## 7. 必须保留的反证与解释边界

| 需要区分的问题 | 目前的证据 | 对主张的影响 |
|---|---|---|
| 能否表示细胞？ | Forebrain profile BA 0.8682；红系模块阶段ρ 0.8322 | 支持“可用表示”，限于相应任务和协议 |
| 是否来自预训练而非架构？ | 完整Forebrain归因：epoch11 BA 0.8682，random42 0.8963；random也有context effect | 尚不支持普遍的预训练优势；一个random seed也不替代全部随机对照 |
| 是否有强输入之外的增量？ | Forebrain B0 BA 0.9099，＋calibrated 0.9096，＋shuffle 0.9111 | 当前未建立稳定独立增量 |
| 是否具有通用扰动响应意义？ | Traxler完整输入rep2：distance AUROC 0.510098；B0 0.555278，＋distance 0.545328 | 不支持通用扰动识别优势 |
| 是否越小越好？ | Adapter-only大幅收缩，但pooled U/S邻域一致性未提高 | 绝对尺度不足以判断表示质量 |
| 是否越大越有动力学意义？ | 固定token效应的方向随gene改变；阶段关联不等于动力学真值 | 不命名为velocity、方向、不稳定性或异常概率 |

“包含细胞状态信息”与“提供表达之外的新信息”是不同强度的命题。Distance是模型输入的确定性读出，不会凭空创造信息；它可能重组输入信息，使某些关系更易被特定任务读取。新增实用价值应通过强基线增量检验，而不能由“context dependence”直接推出。

参考：[预训练归因](figures/pretraining_epoch_distance_multilevel/07_distance_vs_downstream_performance.png) · [Forebrain增量源表](tables/forebrain/incremental_metrics.csv) · [Traxler完整输入报告](reports/full_benchmark_REPORT_CN.md)。

## 8. 围绕这个主题组织正文与图

| 叙事位置 | 需要回答的问题 | 建议已有图/内容 | 不能暗示的结论 |
|---|---|---|---|
| 方法图 | 差异如何变成细胞表示？ | 双模态token→contextual hidden→逐gene cosine→cell profile；对照pooling与scalar | cosine是预训练直接优化目标 |
| 主结果图 | 这种表示是否可用？ | Forebrain跨sample分类；scalar/shuffle/pooled/强输入对照；辅助UMAP | UMAP好看等于性能最优 |
| 解释图 | 差异由目标值还是背景决定？ | 固定token distance实例＋全panel复现＋input/random对照 | 排除所有表达混杂、证明因果调控 |
| 生物支持图 | 能否关联真实阶段？ | Erythroid27样本模块评价，完整保留输入基线 | 预测velocity或发现表达之外调控 |
| 适配图 | 对齐后差异和用途如何变化？ | 多层级distance＋任务/重建/几何随epoch变化 | 距离越小越成功，或对齐必然有害 |

主张优先级：**差异可以作为表示 > 差异受背景调制 > 差异可关联阶段 > 适配改变其用途与结构。** Random、覆盖和强输入基线是解释主发现所必需的证据，不应只因不利而隐藏到没有讨论的附录中。

这是一条研究叙事，不是对现有新颖性或ICLR接收概率的保证。单独的余弦公式并不足够；仍需说明为何选择这种读出、其适用范围，以及相比已有输入和表示具体提供了什么。

## 9. 可直接用于讨论的总结

我们研究的不只是如何融合未剪接与已剪接RNA，而是两者的表征差异能否直接用于表示细胞。scUS将同一细胞、同一基因的U/S contextual embeddings之间的余弦距离组成逐基因profile。现有结果表明，这种差异模式能够区分细胞状态，并通过模块读出关联发育阶段；压缩为单一distance后，其状态识别能力明显下降。即使固定目标基因及U/S离散数值，distance仍随细胞背景变化，说明该读出不只由目标token本身决定。进一步的适配分析显示，缩小U/S距离、改善重建和提高分类并不总是同步发生。上述发现支持将跨模态差异视为可利用的表示，而不是预先认定需要消除的噪声；但其相对随机模型的预训练收益、强表达基线之外的独立增量和更具体的动力学含义仍未建立。

建议当前保留的一句话结论：

> **细胞不仅可以由两个模态各自的表征来描述，也可以由它们在不同基因上的差异模式来描述。**

## 10. 与现有文档的关系

- 本页：围绕“差异也是表示”的当前主线、关键结果与解释，不确定最终题目。
- 完整证据库与旧摘要（未随本结果包发布）：保留更多历史协议、病例、多数据集结果及原摘要，不能把其中旧题目视为本轮定稿。
- 最近结果速览（未随本结果包发布）：查看其他已完成分析分支。
- 结果溯源（未随本结果包发布）：追踪历史图表和数据来源。

本页数值按现有CSV/JSON及报告核对，未启动训练或更改任何实验产物。本发布副本使用包内相对链接；未包含的历史上下文入口已明确标注。它是研究结果快照，不是匿名投稿附件。

# Pretrain与不同Fine-tune策略的U/S distance多层级变化

## 第一入口

- [总览](../figures/us_distance_multilevel_finetune/00_multilevel_distance_overview.png)
- 绝对distance（未随本结果包发布）
- 相对Zero-shot的配对变化（未随本结果包发布）
- 几何保持指标（未随本结果包发布）
- gene-level散点（未随本结果包发布）
- cell×gene密度图（未随本结果包发布）
- cell×gene热图（未随本结果包发布）
- Joint epoch轨迹（未随本结果包发布）
- 不同cell type的cell-level变化（未随本结果包发布）

## 分析定义

所有结果使用相同 **1,720个Forebrain细胞 × 10,692个基因**，共有 **10,865,467** 个有效U/S pair，四种模型的缺失mask完全一致。

- **Cell level**：一个细胞内所有有效基因distance的中位数。
- **Gene level**：一个基因在所有有效细胞中distance的中位数。
- **Cell × gene level**：原始 `1-cos(hU,hS)`，一个观测对应一个细胞中的一个基因。
- **Change from Zero-shot**：先对相同cell×gene pair相减，再在cell或gene内部取中位数；不是两个独立总体中位数之差。

## 主要结果

### 绝对尺度

三层级的中位数如下：

| 模型 | Cell | Gene | Cell × gene |
|---|---:|---:|---:|
| Zero-shot | 0.8566 | 0.8532 | 0.8558 |
| Adapter-only | 0.0036 | 0.0035 | 0.0036 |
| Joint epoch29 | 0.2326 | 0.2282 | 0.2326 |
| Reconstruction-only epoch29 | 0.8720 | 0.8723 | 0.8728 |

Adapter-only几乎把所有U/S向量压到很小角度；Joint也产生明显收缩；Reconstruction-only总体保持Zero-shot的distance尺度。**更小不表示更正确**，只说明表示空间中的U/S角度更接近。

### 相对Zero-shot的变化

Cell-level配对变化中位数：Adapter-only为 **-0.8527**，Joint为 **-0.5425**，Reconstruction-only为 **+0.0173**。

Reconstruction-only没有全局收缩，但gene-level和cell×gene-level仍同时存在增加和降低，说明“总体尺度接近”不等于每个配对都保持不变。具体保持程度见Spearman与median absolute change热图。

## 如何解读三种策略

- **Adapter-only**：Transformer冻结，adapter使绝对distance接近0；这是强烈的几何收缩，不能称为更好的生物对齐。
- **Joint epoch29**：对抗对齐和重建共同训练，保留部分相对结构但明显改变绝对尺度；它对应任务适配与原几何保存之间的折中。
- **Reconstruction-only epoch29**：绝对尺度和多层级结构最接近Zero-shot，说明mask重建更倾向于保留预训练关系几何。
- **Joint epoch49**：只在epoch专图中与相同Joint目标的epoch29比较；它不进入四策略主比较。

## 限制

- 四个模型的输入一致，但训练目标、是否冻结Transformer和历史批次日程不同，因此这是策略级描述，不是单一变量因果实验。
- Cell/gene汇总使用中位数，会隐藏cell×gene层面的异质变化，因此三层图必须一起看。
- cell×gene分布和密度图使用固定随机样本用于绘图；cell和gene层级使用全部观测。
- distance是无符号表征角度，不是RNA velocity、表达方向、异常概率或平衡程度。
- 这些结果描述的是逐基因distance；不要与pooled U/S cell embedding的邻域重合率混淆。

## 可复现性

脚本：scripts/plot_us_distance_multilevel_finetune.py（未随本结果包发布）。输入哈希见input_hashes.json（未随本结果包发布），验收见validation.json（未随本结果包发布）。

## 对应的预训练checkpoint轨迹

不同预训练epoch的相同三层级分析已单独整理，固定使用同一批1,720个细胞和10,692个稳定覆盖基因：

- 预训练epoch总览（未随本结果包发布）
- 绝对distance轨迹（未随本结果包发布）
- [distance与下游性能](../figures/pretraining_epoch_distance_multilevel/07_distance_vs_downstream_performance.png)
- 完整中文报告（未随本结果包发布）

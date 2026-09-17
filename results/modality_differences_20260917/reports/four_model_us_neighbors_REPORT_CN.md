# 4 模型 U/S 邻域一致性与 k 敏感性

总览（未随本结果包发布）

## 输入与方法

复用原三模型缓存，只补充 reconstruction-only epoch29 编码，不重新训练。相同 1,720 细胞、完整历史联合 U/S 输入、10,692 基因面板。模型：zero_shot, adapter_only, joint_epoch29, reconstruction_epoch29。
每细胞 U/S 分别在相同有效配对基因上平均，得到两个128维向量。缺失配对保留NaN，真实零距离保留。
U/S各自逐维z-score后RMS为主度量，原始cosine作敏感性。计算k=[5, 10, 15, 30, 50]，重点展示[10, 30, 50]；排除自身，并列按cell ID排序。
**overlap=共同邻居数/k；Jaccard=共同邻居数/(2k−共同邻居数)**，比较同模型U和S，不是相对Zero-shot的邻居保持率。
模型配对差值为右模型一致率减左模型一致率。均匀随机参照k/1719随k变化，不能直接把更大k下更高一致率当成改善。

## 主结果：标准化 RMS

| model | 10 | 30 | 50 |
| --- | --- | --- | --- |
| zero_shot | 0.47541 | 0.57047 | 0.61299 |
| adapter_only | 0.40837 | 0.49450 | 0.53620 |
| joint_epoch29 | 0.43703 | 0.52816 | 0.57238 |
| reconstruction_epoch29 | 0.49622 | 0.58926 | 0.63388 |

## 敏感性：原始 cosine

| model | 10 | 30 | 50 |
| --- | --- | --- | --- |
| zero_shot | 0.38552 | 0.46990 | 0.50736 |
| adapter_only | 0.35587 | 0.43820 | 0.47667 |
| joint_epoch29 | 0.37203 | 0.45981 | 0.50571 |
| reconstruction_epoch29 | 0.41512 | 0.49203 | 0.53150 |

## 同细胞配对变化

| left | right | k | mean_delta | fraction_increased | fraction_equal |
| --- | --- | --- | --- | --- | --- |
| adapter_only | joint_epoch29 | 10 | 0.02866 | 0.45291 | 0.22093 |
| adapter_only | joint_epoch29 | 30 | 0.03366 | 0.55756 | 0.11860 |
| adapter_only | joint_epoch29 | 50 | 0.03619 | 0.59535 | 0.08837 |
| adapter_only | reconstruction_epoch29 | 10 | 0.08785 | 0.59244 | 0.21686 |
| adapter_only | reconstruction_epoch29 | 30 | 0.09477 | 0.79244 | 0.07791 |
| adapter_only | reconstruction_epoch29 | 50 | 0.09769 | 0.86047 | 0.04535 |
| joint_epoch29 | reconstruction_epoch29 | 10 | 0.05919 | 0.52151 | 0.23779 |
| joint_epoch29 | reconstruction_epoch29 | 30 | 0.06110 | 0.67616 | 0.11686 |
| joint_epoch29 | reconstruction_epoch29 | 50 | 0.06150 | 0.73488 | 0.07151 |
| zero_shot | adapter_only | 10 | -0.06703 | 0.21628 | 0.24942 |
| zero_shot | adapter_only | 30 | -0.07597 | 0.16453 | 0.10640 |
| zero_shot | adapter_only | 50 | -0.07679 | 0.14302 | 0.05872 |
| zero_shot | joint_epoch29 | 10 | -0.03837 | 0.29535 | 0.24070 |
| zero_shot | joint_epoch29 | 30 | -0.04231 | 0.27384 | 0.12674 |
| zero_shot | joint_epoch29 | 50 | -0.04060 | 0.27093 | 0.08488 |
| zero_shot | reconstruction_epoch29 | 10 | 0.02081 | 0.39884 | 0.33953 |
| zero_shot | reconstruction_epoch29 | 30 | 0.01880 | 0.49593 | 0.20233 |
| zero_shot | reconstruction_epoch29 | 50 | 0.02090 | 0.57500 | 0.13547 |

## 固定掩码重建评价

| model | evaluation | loss_rec | accuracy_u | accuracy_s |
| --- | --- | --- | --- | --- |
| adapter_only | frozen_head_readout_compatibility | 2.75068 | 0.06618 | 0.06768 |
| joint_epoch29 | native_reconstruction | 2.04531 | 0.22655 | 0.29062 |
| reconstruction_epoch29 | native_reconstruction | 2.02621 | 0.22919 | 0.29400 |
| zero_shot | native_reconstruction | 2.09517 | 0.22143 | 0.28466 |

原132内部验证细胞，seed42、mask epoch0、两个历史重复；掩码比例读取checkpoint，各模型逐cell/replica掩码和计数相同。
CE先在每细胞内U/S等权，再对重复、细胞等权，不乘训练权重。adapter接冻结预训练头仅为读取兼容性，不参与公平重建排名。

## 结构与解释边界

Reconstruction-only 使用已完成的 epoch29：adv=0、rec=1.0；Joint epoch29 使用 adv=0.1、rec=0.3。
Reconstruction-only 30 epochs 均为 batch32；历史 Joint epoch0–5 为 batch1、epoch6–29 为 batch32，对抗项另有五轮 warm-up。
两者更新 encoder、residual adapter 和重建头，原预训练起点相同，但权重、批次日程不同；本结果是训练方案比较，不是 Align 单因素因果消融。
旧 adapter-only 冻结 encoder，且见过全部细胞。所有细胞共同参与标准化和邻域构建，因此是描述性诊断，不宣称公平外层泛化。

同cell-type比例、同类型邻居中的跨sample比例分别报告U和S；无同类型邻居时条件比例为NA。
有效秩使用中心化原始向量，另存标准化有效秩。高U/S一致率须结合这些结构指标看，不能单独证明生物学改善。
逐基因U/S距离为1−cos(U,S)，与pooling后细胞间距离、重建CE分开保存；它不是RNA velocity。
Spearman仅描述同模型及cell-type内关联，常数或不足3个观测为NA，不解释为因果。
8幅UMAP独立拟合，seed42、15邻居、min_dist0.5；不同k只换一致率颜色，不换坐标，不把坐标差解释为细胞移动。
两侧来自同一次联合编码，不代表U-only与S-only独立输入一致性。

## 图表与数据

- 00_overview（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 01_agreement_k10（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 01b_strata_k10（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 02_quality_k10（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 02b_composition_k10（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 02c_metric_sensitivity_k10（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 03_cell_type_umaps（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 03_overlap_umaps_k10（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 04_distance_agreement_k10（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 04b_correlations_k10（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 01_agreement_k30（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 01b_strata_k30（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 02_quality_k30（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 02b_composition_k30（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 02c_metric_sensitivity_k30（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 03_overlap_umaps_k30（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 04_distance_agreement_k30（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 04b_correlations_k30（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 01_agreement_k50（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 01b_strata_k50（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 02_quality_k50（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 02b_composition_k50（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 02c_metric_sensitivity_k50（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 03_overlap_umaps_k50（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 04_distance_agreement_k50（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 04b_correlations_k50（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）
- 04c_reconstruction（未随本结果包发布） · PDF（未随本结果包发布） · CSV（未随本结果包发布）

各图PNG/PDF/CSV齐全。完整U/S细胞间距离和最多50邻居索引/距离位于各模型/度量目录。
consistency_cells.csv为逐细胞指标；quality_by_*.csv与consistency_by_*.csv为分层结果；paired_model_changes.csv为同细胞配对。
输入缓存位于 <WORKSPACE>/scUS/outputs/paper/four_model_us_neighbors/cache，保存编码、固定掩码评价、输入身份和smoke。input_hashes.json记录输入哈希。
historical_regression.csv核对原三模型k5/15/30逐细胞结果；validation.json记录验收。

## 复现

```bash
python scripts/three_model_us_neighbors.py all --input-dir <WORKSPACE>/scUS/outputs/paper/four_model_us_neighbors/cache --output-dir <WORKSPACE>/scUS/outputs/paper/four_model_us_neighbors --model-list <WORKSPACE>/scUS/outputs/paper/four_model_us_neighbors/models.json --ks 5 10 15 30 50 --display-ks 10 30 50
```

如需补齐仅重建模型缓存，运行 `python scripts/four_model_us_cache.py all --device cuda:3 --unrestricted-gpu`。
新增模型编码使用BF16 encoder、FP32 adapter/head，batch32；CPU线程4。用户于2026-09-15取消GPU资源等待门槛和进程显存上限，本次直接在GPU3执行；策略记录见cache/resource_policy.json。
邻域与绘图在CPU完成；原三模型结果不覆盖。分析脚本默认调用仍支持原三模型、k5/15/30。

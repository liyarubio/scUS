# Forebrain 条件校准与公平留出评估

参考 sample 为 `10X_17_029`，所有校准、基因筛选、标准化及增量模型均只在该 sample 拟合；`10X_17_028` 仅用于一次外层测试。

## 15-NN profile 结果

| representation | balanced_accuracy | macro_f1 | macro_auprc |
|---|---|---|---|
| raw_distance_profile | 0.8682 | 0.8869 | 0.9589 |
| calibrated_distance_profile | 0.8925 | 0.9051 | 0.9645 |
| direct_bin_difference | 0.9052 | 0.9168 | 0.9691 |
| absolute_bin_difference | 0.9035 | 0.9150 | 0.9707 |
| pair_mask | 0.9018 | 0.9112 | 0.9680 |
| distance_scalar | 0.3492 | 0.3515 | 0.3580 |
| gene_shuffled_distance | 0.2464 | 0.2269 | 0.2498 |
| state | 0.8604 | 0.8733 | 0.9351 |
| u_only | 0.7673 | 0.7903 | 0.8868 |
| s_only | 0.7649 | 0.7842 | 0.8873 |
| concatenated_u_s | 0.7828 | 0.8025 | 0.8960 |
| signed_kinetic | 0.6938 | 0.7101 | 0.7801 |
| absolute_kinetic | 0.7999 | 0.8209 | 0.9023 |
| continuous_moments_difference | 0.9034 | 0.9141 | 0.9683 |
| continuous_moments_abs_difference | 0.9043 | 0.9154 | 0.9684 |
| continuous_moments_total | 0.9025 | 0.9128 | 0.9701 |

## 增量模型

| model | selected_C | features_after_transform | balanced_accuracy | macro_f1 | macro_auprc |
|---|---|---|---|---|---|
| B0 | 0.1000 | 640 | 0.9099 | 0.9173 | 0.9616 |
| B1_calibrated | 0.0100 | 768 | 0.9096 | 0.9157 | 0.9665 |
| B0_plus_gene_shuffle | 0.0100 | 768 | 0.9111 | 0.9186 | 0.9689 |
| B0_plus_random | 1.0000 | 768 | 0.9073 | 0.9152 | 0.9602 |

`calibrated distance` 是条件残差，不是异常概率。该数据只有两个 sample，内部分类器选择仍是细胞级交叉验证，不能视为独立生物重复。

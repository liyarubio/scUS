# 完整输入Traxler验证

按replicate分别重建moments；全部来源预处理后词表有效基因、无top-k。模型为固定epoch11以及3个随机初始化同架构。不含任何Align或重新预训练。

| method | rep2 AUROC |
| --- | --- |
| input_embedding_mean | 0.538062 |
| abs_moments_delta | 0.532625 |
| u_only | 0.530702 |
| detection_qc | 0.524070 |
| abs_bin_delta | 0.522906 |
| bin_delta | 0.520539 |
| pair_mask | 0.519817 |
| absolute_kinetic | 0.513677 |
| concatenated_u_s | 0.513224 |
| log_total | 0.511959 |
| moments_delta | 0.511618 |
| distance_profile | 0.510098 |
| random44_distance_profile | 0.505513 |
| random43_distance_profile | 0.505204 |
| state | 0.504580 |
| random42_distance_profile | 0.503857 |
| s_only | 0.503216 |
| distance_scalar | 0.494720 |
| signed_kinetic | 0.490916 |
| count_delta | 0.486141 |

| method | rep2 AUROC |
| --- | --- |
| B0 | 0.555278 |
| B0+distance | 0.545328 |
| B0+random | 0.558390 |
| B0+shuffled | 0.544607 |
| state | 0.529195 |
| state+distance | 0.506142 |
| state+random | 0.522386 |
| state+shuffled | 0.505465 |

全输入C仅由rep1选择；rep2不用于选特征、模型或任务。历史rep2曾被分析过，本轮也有缓存对照，因此不是此前从未查看的盲测。仅有两个replicate，预训练暴露未知。

# 历史资产审计

状态：完成只读资产核验；尚无本周性能结论。

已核验 21 个必要历史权重、4个真实TRAIN张量数据集及7个共同Stage1缓存索引。无20k重训、无大型数据下载。完整路径、SHA、缺失与复用类别见 REUSED_EVIDENCE_REGISTRY.json。

| 历史方法 | HOTA | 边界 |
| --- | ---: | --- |
| VII fixed rule | 85.03 | B2 84.9749; original heldouts09/10/11, not current main table |
| XIII Fixed Question | 75.637 | Full75.056; different learned WHO protocol |
| XIV Multi | 73.505 | AssA73.639/CVIDF1 88.546/IDSW391.667; seed20261009 IDSW824 retained |
| XV V3 | 74.188 | AssA74.891/CVIDF1 90.286/IDSW147; pilot diagnostic |
| XVI hard memory | 61.764 | IDSW1494; rejected |
| Original full GMT | 78.11 | Separate trained detector/VFCE/RPCE path; never called same frontend |

上述不同阶段指标不可拼成同条件主表。MLP没有对应20k正式权重，不能用历史弱/失败MLP替代公平新控制。原始GMT的1152D旧TRAIN缓存仍在，可尝试真实原生恢复；它与1024D Stage1共同缓存的差异必须单列。

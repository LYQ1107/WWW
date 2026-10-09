# Identity Error Taxonomy

冻结证据：`reports/JEV_PHASE14/PHASE13_FROZEN_EVIDENCE.json`。

所有 GT 仅在保存预测后、离线通过与原dense标签一致的逐图 Hungarian IoU>=0.5 对齐。身份锚点只使用当前提交前的历史观察；未知检测不投票，混合历史不使用多数投票。

| 类型 | 可验证定义 |
|---|---|
| WRONG_EXISTING_MATCH | 选择纯净单GT已有身份，其历史GT与当前认证GT不同 |
| FALSE_DEFER | MATCH拒绝，但合法active集合有纯净正确身份；后续恢复成功也单独记录 |
| FALSE_SPLIT | 上述状态最终START_NEW |
| FALSE_MERGE | 既有已锚定ID第一次写入另一个GT；污染延续另外统计 |
| WRONG_REACTIVATION | REACT选择纯净stale身份，其GT与当前GT不同 |
| FALSE_BIRTH | 当前GT在任何已提交摄像头历史出现过，又START_NEW |
| GALLERY_CONTAMINATION | 当前提交向已有ID首次引入不同GT；与所有污染状态写入次数分开 |
| IDENTITY_UNAVAILABLE | 所有合法候选的观察历史均可认证且无正确身份；有任何UNKNOWN则可用性UNKNOWN |
| UNKNOWN_SUPERVISION | 当前GT或候选历史无法认证，不能强制判错 |
| CROSS_CAMERA_ID_MISMATCH | 同GT在最近40帧另一相机观察到不同GlobalID；逐观察量，非CVIDF1 |

事件类型可以重叠。CLEAR IDSW按官方上一匹配ID变化独立统计并逐项重建；每次IDSW只分配一个优先归因。错误事件、轨迹混合数、出生数、碎片数、IDSW和观察持续时间不得互相替代。

# Phase XV UNKNOWN and Reliability

UNKNOWN 保留为合法推理候选，不自动作负例或正例。WHO 的已认证纯净身份兼容性、具体 Commit 的可安全延续性、全局 history purity、预测 safety 与 uncertainty 分开。

V1 all-query audit:3140个当前 GT已知查询中1309选择 UNKNOWN，995选择全局混合历史；WHO认证查询仅1822。认证子集接近满分不能代表其余困难查询的风险。V2 在3160个已知查询中1310选择UNKNOWN，990选择混合历史；不采用未认证情况的猜测真值。两版各自选择认证错误11和12，UNKNOWN 不计为已认证错误。

V3 全部保留查询计数（其审计 corpus 与前两版不同，不能直接把绝对计数当改进）：

```json
{
  "DEFER_all_rows": 0,
  "WHO_certified_query_rows": 1749,
  "all_query_rows": 3324,
  "current_GT_UNKNOWN_rows": 199,
  "current_GT_known_rows": 3125,
  "joint_selected_differs_row_argmax": 116,
  "mixed_selected_despite_pure_alternative": 0,
  "no_certified_pure_correct_candidate_queries": 1376,
  "pure_supported_queries": 1749,
  "selected_UNKNOWN_all_GT_known_rows": 1381,
  "selected_UNKNOWN_uncertified_queries": 1375,
  "selected_certified_correct_all_GT_known_rows": 1743,
  "selected_certified_wrong_all_GT_known_rows": 1,
  "selected_globally_mixed_all_GT_known_rows": 1032
}
```

Risk-coverage 的 coverage 分母采用全部当前 GT已知查询，certified-selection risk仅在被选候选已有认证时计算，UNKNOWN数量另列。当前 GT未知的检测另列 unassessed。置信度是实际联合分配选中动作的softmax概率；不能用单行最大分替代。在V1/V2分别有201/157个查询的联合动作不同于单行argmax；旧最大概率口径保留，SELECTED_ACTION_RISK_V1/V2修正副本分别绑定不变的checkpoint，没有额外SGD。概率未经独立校准，高置信UNKNOWN不等于正确。

V2 recent3-frame 认证只给“该候选当前局部承诺不安全”提供证据；未知/缺帧、跨帧不连续、最近属于当前人、没有纯净兼容替代，都不制造纠错标签。WHO对混合历史仍UNKNOWN。更一般的无纯净候选状态不能随意补充正确候选或宣称 DEFER一定安全：P2 DEFER实际常造成假出生和错误混合。

完全保留 annotation，重复 GT身份仅使离线认证/风险观察 UNKNOWN；TrackEval及官方 MATLAB仍读取原始GT。训练不能用DEV结果挑门槛。真实完整视频风险观察、错误时长与CVIDF1独立报告。

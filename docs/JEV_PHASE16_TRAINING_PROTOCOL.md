# Phase XVI 训练资格与停止协议

本轮训练不是默认执行步骤。必须先有合格历史证据、原生安全记忆收益及真实可区分监督。P1 采用冻结的 +2pp 支持、≤1pp 错激活增加、至少八个独立新支持 cluster 门槛；不因基线已饱和而降低门槛。独立训练的 reserved TRAIN temporal block 固定为 `(scene_frame//64)%5==4`，DEV17/18/19 不参与规则、阈值、标签选择。

安全恢复正例要求改变实际首次决策、将已认证错误 owner 改为正确 owner，并且所有预定 H 上安全且有真实未来收益。未改变的延续、纯影子相似度支持、UNKNOWN 转移或同人物的另一个未认证碎片不能伪造具体 Global ID 恢复标签。不安全恢复负例也必须来自实际改变动作及错误 owner/有害原生后果；记忆规则后果不能直接冒充某个 JEV 动作的监督。

至少需要八个独立安全恢复 cluster、八个不安全 cluster，TRAIN 与 reserved 有代表性，并且 reserved 必要纠错至少八个 cluster；保留正常连续关联，不用总 accuracy 掩盖必要纠错稀缺。任一前置门槛不通过时，Tiny/Pilot/Formal/Matched4k 保持 NOT_RUN，指标 null。

合格时先 Tiny128，再 Pilot1500 seed20261009；只有安全原生 Pilot PASS，才允许 20k×20261008/09/10 与同证据同数据同预算 Fixed Question / Set Transformer。正式和真实闭环合格后才考虑 matched4k。三种子完整评价之前不宣称已消除历史最差种子失稳，也不以单种子低 IDSW 证明动态 JEV 独立价值。

本轮最多三个主要科研版本。V1 冻结证据读取，V2 安全记忆，V3 是有条件的学习版本；V3 无资格时不通过修改网络容量或增加训练步数强行启动。工程修复和精确重放不计新假设，但原始尝试、日志和哈希全部保留。SEALED20/21/22、official TEST 和 Full24 均不授权。

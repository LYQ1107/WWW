# Dense reliable action supervision

Frozen evidence: `reports/JEV_PHASE14/PHASE13_FROZEN_EVIDENCE.json`。

复用TRAIN12/13/14/16的既有真实B1-native输入。用保存的cosine值与完整合法Hungarian重建每次active/stale选择与新增ID，全部原认证candidate labels和最终ID计数逐项一致，GT只构造离线标签。

自然Q2只认证明确纯净正确候选存在，或所有合法候选均有可认证历史且无正确身份。其它状态保持UNKNOWN。Q3需要>=3已知观察和>=80%已知覆盖；污染要求至少两个GT各有>=2观察。标签代表可确认的过去观察，无法证明未标注部分的真正身份纯净。

三个输入副本：自然presence、物理删除全部纯净正确候选的absence、删除同样数量认证错误候选的count-matched presence。其余视觉/时间/相机/历史来源不变，UNKNOWN候选存在时不制造缺席证书。

按video与64帧时间块采样，block编号%5==4只作TRAIN审计不接收梯度。样本是各自payload完整联合分配。对比CE、旧structured loss、成本敏感UNKNOWN-safe完整联合margin、多任务availability/trust联合目标；所有结构共享相同控制矩阵。

UNKNOWN-safe margin将可认证正确选项与所有合法UNKNOWN都放入零代价可接受集合。损失增强保留全部合法候选，惩罚明确已认证错误，不能把UNKNOWN选中本身当错误。代价2来自预注册对合并/分裂的对称先验，没有查看新训练HOTA后修改。

MATCH terminal是DEFER；真正native START_NEW发生在随后冻结bank fallback之后，两者分别记录。没有自然支持时不训练NEW/REACT或伪造memory action reward。

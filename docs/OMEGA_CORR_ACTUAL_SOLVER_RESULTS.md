# Ω-Corr 原生求解状态

NOT_EXECUTED：固定GateB未通过，因此没有训练校准头，也没有将Ω-Corr插入get_asso或执行新状态分叉。Hungarian、strict threshold、新ID、Bank与最终身份变化均为null/未测，不能写成零收益。

160组原生上下文重建仅校验OFF排序和既有实际匹配分数，不是Ω-Corr真实求解实验。原生缓存重建涉及8,456次新增图像forward、242张预检复用和552张封存Scene12上下文复用；309次主关联调用，共1,895查询，旧缓存raw矩阵与实际已选分数误差0。原来的Ω-F求解负结果单独保留，不能作为本轮失败证据。

行排序净纠错为负且没有独有证据，不绕过门槛运行新solver/fullvideo。机器状态见omega_corr_solver_decisions.json。

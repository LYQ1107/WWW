# 完整 GMT 基线审计

使用冻结项目源04324354的原始完整GMT（旧JEV全部关闭），Stage2 checkpoint `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`严格加载411个tensor，无missing/unexpected键。WITH_CAT=true，CAT_DIM=128：真实VFCE1024+RPCE128，GTA1152；s_t_head输入11，输出128，GTA实际执行。全部参数53515309，未剪除GTA/RPCE，未替换REACT/WRITE。

前32帧同图像、同权重、同随机状态的原始源码快照和Adapter OFF完全一致：GTA logits、聚合分数、native IDs、Gallery/Bank状态摘要与tensor内容、完整原生预测。原代码快照从Git精确提取到本轮runtime；比较对象是已有项目冻结源码，包含历史工程兼容修正，不伪称新的官方byte-identical源码复刻。

输入为真实图像，包含Stage2训练的检测/VFCE/RPCE，不复用旧Stage1-only缓存。全视频parity与匹配评测仍待完成；当前科学状态ACTIVE，旧78.110等DEV分数不移植到TRAIN16。带hash诊断的运行时间不是完整FPS。

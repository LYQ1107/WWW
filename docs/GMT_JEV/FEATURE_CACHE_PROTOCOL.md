# 可选历史归一化缓存：等价性与效率协议

固定 Rule 的分数公式、risk gate、合法候选、全部 GMT 权重和原生求解器。每次实际提交更新当前摄像头的归一化历史和两摄像头历史一致性缓存，当前 age、候选和竞争信息仍重新读取。缓存是已提交历史的派生量，不能含 GT 或未来信息。

已有所有控制器默认 cache_features=False，既有质量结果和当前计时任务保留。可选 cached Rule 是同一决策公式的工程实现，不是新阈值或新模型。暂不据此宣称学习式优势。

在原七项同卡计时队列完成后，GPU7 执行完整 TRAIN16 的 paired shadow 审计：cached/uncached 接收同一真实原生状态，要求全部24特征、eligibility、gate、residual、corrected scores 逐项 torch.equal，全部1034场景的 raw predictions 还须与此前已保存 Rule 运行完全相同。任何失败不得启动 cached FPS。

通过后才另做 cached Rule 三次真实完整图像计时，包含相同 stock 后处理。所有原计时结果留存，cached 单独报告；不能按正号切换 raw/stock 主次约定。无新训练、无权重写入、无数据下载、无新DEV/封存/TEST访问。原12 GPU-wall-hours、512 MiB及30 GiB保留空间门槛继续覆盖新增工作。

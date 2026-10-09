Phase XIV 用两条独立验证方案区分“未见数据集迁移”与“全部预训练都未暴露的完整系统泛化”。冻结证据是 reports/JEV_PHASE14/PHASE13_FROZEN_EVIDENCE.json；独立数据协议在读取外部身份标注与追踪指标前已冻结。

实际 Stage1 与原始 GMT Stage2 的训练 JSON 都包含 VisionTrack 的全部 24 个 TRAIN 视频、57,508 张图像。关联控制器使用 TRAIN12/13/14/16，开发视频为17/18/19；20/21/22仍封存。数字 instanceID 相等不能证明跨独立视频的同一生物身份。原前端下的开发验证不能称为未见场景的完整系统测试。

方案 A 固定使用 EPFL 官方 WILDTRACK 归档的 C1/C2，按文件名取前320个真实标注时刻。官方页面说明其为 ETH Zurich 的七相机采集；[GMT 论文](https://arxiv.org/html/2407.01007v2)说明 VisionTrack 为双移动无人机采集。[WILDTRACK 官方页面](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)与实际 JSON 核验支持跨时间 personID；所选320帧有276个不同personID，其中265个跨帧出现。仅下载相应640张真实图像与320份标注，不插值检测或GT，不读取其他相机图像及后80帧标注。共享personID JSON天然包含七视角投影：这些共享文件完整读取用于schema审计，实际评估只使用viewNum0/1。HTTP块边界可能含相邻归档压缩字节，其他成员未解码。

所有模型、输入阈值、相机编号、时间前缀和评估规则在外部追踪指标出现前冻结。外部GT只用于预测提交后的评估，不训练、校准或选择checkpoint。比较旧Full/Fixed/Set、Cosine及本阶段全部正式LAST模型；三种子结果全部报告。此结果是固定两相机、160秒、2FPS观察采样的迁移实验，不能当作完整七相机WILDTRACK官方benchmark。原生窗口保持观察帧单位，因此与30FPS VisionTrack存在明确的物理时间尺度差异。

方案 B 必须从可核验的通用视觉初始化重新训练完整Stage1，并为GMT和全部关联对照生成同一新前端、执行公平Stage2训练。目前可用CH_FPN_1x及其key_adapted文件的张量内容多重集合逐位相同，但checkpoint没有可核验的训练数据清单，作者下载链接也没有公开匹配这些实际文件的内容校验值。文件名CH不能认证CrowdHuman训练，更不能认证未接触全部评估场景。故方案 B 暂未通过初始化、独立预算与配置资格，不启动昂贵重训；不会把旧Stage1改名为clean模型。

方案 A 可以回答已知VisionTrack训练清单之外的数据集迁移；全部预训练无暴露的更强完整系统结论仍为NO_GO。此限制在PRETRAIN_EXPOSURE_AUDIT.json中保留，不通过重新定义heldout解除。官方TEST、Full24和封存视频继续禁止自动启用。

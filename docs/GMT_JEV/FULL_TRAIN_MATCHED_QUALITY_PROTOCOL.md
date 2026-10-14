# 完整 TRAIN 轨迹配对评测（执行前冻结）

严格 accepted-reference 计数不能覆盖原始拒绝关联后的 Bank/新生身份反事实。受限 ORACLE 中也存在这一 UNKNOWN 覆盖。因此 N01=0/N10=1 仅证明该认证提交子集未呈现净收益，不能单独推导整个 Rule 轨迹完全无收益。本检查补齐 TRAIN12/13/14 的 Original GMT 与相同冻结 Rule 冷启动完整评测，不增加架构、规则系数或数据范围。

三段视频、两种方法全部报告，先 Original 后 Rule，固定20261009种子。Original 使用既有原始 GTRRCNN 来源快照；Rule 使用已审计的原生插入点。完整检测/VFCE/RPCE/GTA、候选生成规则、solver、阈值、Gallery/Bank、WRITE/REACT和原生状态转换不变。两边状态各自真实演化，不强行固定为离线候选回放。Rule 历史摘要只来自实际提交。推理不安装 GT 观察器，不做梯度更新、不缓存 perception；GT 仅在推理结束后评测。

记录所有输入图像的组合内容 SHA、有效配置 SHA 和 raw 几何/置信度 SHA；两种方法必须一致。主要指标继续为保留短轨的 raw，次要指标为原生 min_track_len/stock 后处理，均使用既有冻结 TrackEval 与官方 MATLAB 实现。分别报告每段 HOTA/AssA/DetA/IDF1/CVIDF1/CVMA/MOTA/IDSW/Frag，以及相同约定下的回顾性身份持续量。保留 UNKNOWN/描述性时间与因果传播的区别。

完成六段推理后，在相同三段组成上运行合并 TrackEval/CV计数评测，不能简单平均 HOTA 或挑有利视频。此处是已暴露 TRAIN 上的完整轨迹诊断与后续训练资格参考，不是独立验证、不覆盖 TRAIN16 主结果、不把 Rule 或 JEV 判为科学成功。既有局部/原生计数与 NO_GO 原样保留；任何训练还需合格监督采样和公平协议。

不改变既有16的视频评测调用，扩展评测接口只接受12/13/14/16白名单。新增仅紧凑统计与必要预测压缩包，不保留图像/训练样本/新权重。预计新增约0.8 GPU小时，仍受整体12 GPU-wall-hours、512 MiB留存和home至少30 GiB保留空间约束。无新 DEV、封存视频、TEST或Full24。

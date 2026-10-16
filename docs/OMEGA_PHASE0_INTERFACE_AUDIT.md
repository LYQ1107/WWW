# Ω Phase0 接口审计

416主实验：**BLOCKED_WEIGHT_ACCESS / NOT_EXECUTED**。512仅为单独的固定样本诊断，不替代416，不代表身份或跟踪成功。所有24组及失败样本均保留。

官方源码固定48b23c8c，原始512文件实际SHA与官方LFS值一致；公开分发来源及许可记录见SOURCE_AUDIT和acquisition receipt。1411项state严格加载，全部冻结。24组×AB/BA=48次真实前向均输出有限pose、camera、depth和confidence。实际shape见omega_camera_pose_comparison.csv。外参camera-from-world，confidence=1+exp(logit)，未sigmoid或当成身份概率。

独立torch2.6.0+cu118环境，V100sm70；其默认BF16检测包含模拟并返回True，native-only为False。使用官方autocast=False构造参数+外层FP16，官方fp32相机/深度head guard保留；没有修改官方源码/权重。10项坐标/合成几何/权重拒绝/AMP契约通过，真实24对两种分辨率1986点逆映射最大误差4.55e-13像素。像素中心半像素修正和连续边界矩阵分别记录，裁剪丢失/padding逆映射为invalid。

新旧VGGT均在同一次forward处理物理AB或BA；GMT从未运行或改序。源码SHA、环境freeze、实际命令和完整进程计时均由原始manifest/receipt绑定。

# Ω-Corr 实测成本与条件未执行

所有已运行GPU可见子进程（含失败、导入、读权重、预处理、推理与导出）的累计墙时为0.670985 GPU-visible process-hours；不是CUDA纯kernel时间，也不是GPU利用率积分。第一阶段2小时预算通过。研究缓存975833266 bytes（不含代码、deps及日志）小于2GiB。

完整Ω研究提取800次forward=160联合+320真正单视图+320错搭档控制。另接口5次；旧VGGT成功160+hook校验3，DINO160次双图独立批次；224/336局部crop每个模型686次独立proposal调用。CPU成对统计v1因逐标量反复解压NPZ而主动终止（exit -15），日志与完整墙时保留；v2只将相同数组一次读入内存，分数和种子不变。首次old/DINO patch16误用失败成本也保留，源代码显示各完成首个forward后出错，计数器未导出，不能冒充实测计数。

Ω联合forward的160组P50=0.179506s、P95=0.187242s；包括首次冷forward、不含权重加载。全子进程成本、各模型forward P50/P95、峰值allocated/reserved分别见runtime.csv和原始manifest/receipts。完整研究提取含对照额外调用，不等于默认部署每组一次调用。

FULL_TRACKING_EFFICIENCY=NOT_EXECUTED/NOT_MEASURED。没有GMT原生在线FPS、系统FPS、在线P95、跳过率或10%/20%触发收益；不以缓存读取/离线提取速度冒充实时系统。局部686次crop调用只作有限尺度机制对照，未实现在线调度。模型常驻内存与系统共存开销未在线测量；独立提取峰值不等于系统额外显存。

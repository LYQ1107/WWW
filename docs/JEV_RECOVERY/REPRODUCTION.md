# 复现与运行状态

工作树 /home/liuyeqiang/WWW_jev_recovery_20261012；运行目录 /data1/liuyeqiang/WWW_jev_recovery_runtime/20261012_v1。使用 GMT Python 环境；原始视频/模型/缓存只读，新增数据留服务器。

查询状态：`PYTHONPATH=reproduction_tools /home/liuyeqiang/anaconda3/envs/GMT/bin/python reproduction_tools/jev_recovery_controller.py --status`。EXPERIMENT_REGISTRY.json 内每个 RUNNING 条目提供 PID、GPU、独立日志和源码绑定，训练每32步保存 PROGRESS；native 每10秒保存帧进度。

R1_CONTROLLER.log → 首版 Tiny/Pilot/完整TRAIN16；PROGRESSION.log → 根据 TRAIN 门槛推进R2/R3和合格模型三种子4000步/完整DEV；FINISH.log → 同状态分支、身份指标与同卡完整图像FPS。未完成项不能填数字，失败保留原始日志，控制器遇到失败停止依赖任务供修复。

原一周效率仍独立使用GPU5。本轮GPU6训练；GPU1仅在有显存时叠加有界工程验证。新FPS必须等待GPU6完全空闲，三次完整video17，每次重新加载模型、8帧预热再真实冷启动，无GT评价或journals。

候选方案初始4/8诊断未提高已认证正确身份覆盖；在第一次新训练前冻结8方案以保留原候选。原RUN_V1与原4方案负结果保留，执行冻结见CANDIDATE_EXECUTION_FREEZE.json。

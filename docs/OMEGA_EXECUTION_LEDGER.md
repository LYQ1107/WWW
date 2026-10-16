# Ω 本轮执行与继续入口

当前状态为 **DIAGNOSTICS_COMPLETE_MAIN416_ACCESS_BLOCKED**，主科学目标未完成。
本文件不是一份已执行416或已获得身份收益的声明。

实际执行源为只读使用的detached工作树
`/data1/liuyeqiang/WWW_omega_runtime_20261016/sources/phase0_native_fp16`，提交
`317d6e41`；其源文件SHA另存原始producer manifests。
外部官方Ω源码固定48b23c8c，旧VGGT源码固定a288dd0f。没有修改这两份源码。
新环境 `/data1/liuyeqiang/WWW_omega_runtime_20261016/venv` 的torch为2.6.0+cu118，
torchvision为0.21.0+cu118；完整依赖freeze文件及SHA在storage/setup proof。

首次隔离安装失败于typing-extensions的名称元数据和缺少build依赖，失败日志保留；
仅在新环境升级pip并从PyPI补依赖，再安装精确cu118 wheel成功。
旧GMT环境仍为torch2.0.0+cu118，未进行全局升级。

本次公开512镜像来自1kaiser/vggt-omega-jax的a8c3a718 revision；完整文件
4576706117字节，实际SHA与官方1041e80f revision的LFS值一致。
许可证随本地文件保留，与官方LICENSE blob相同。没有下载GGUF/MLX转换文件，
没有向Git上传权重、原始图像或dense特征。416官方访问仍需账户批准，未绕过受保护端点。

## 实际生产命令

从上面的detached工作树运行，环境为 `CUDA_VISIBLE_DEVICES=0`、
`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`。精确完整参数数组、child PID、
源commit、日志SHA及全进程时长在下面的两个receipt中，也已打包进numeric evidence：

- `logs/omega512_initial24_v1_receipt.json`：调用
  `geo_tools/omega_pose_audit.py --resolution 512 --checkpoint /data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt --tag omega512_initial24_v1 --limit 24`。
- `logs/old_vggt_initial24_v1_receipt.json`：调用
  `geo_tools/omega_old_vggt_pose_audit.py --omega-run /data1/liuyeqiang/WWW_omega_runtime_20261016/pose_audits/omega512_initial24_v1 --tag old_vggt_initial24_v1`。

两条均通过 `geo_tools/omega_run_measured.py` 启动，计入整个child导入、输入/权重读取、
构建、预处理、前向、检查和导出。各运行24个物理图像对、48次模型调用，两个进程均正常结束。
没有重跑已完成GMT全TRAIN审计或旧实验；旧VGGT此次在新固定图像对上进行新的对照。

Ω完整进程100.509460703秒，旧VGGT81.810607485秒，共0.0506444634 GPU-visible-process-wall小时。
Ω还承担SIFT提取/初步传统F/小对应点导出，旧VGGT复用这些点；旧模型额外产生world_points。
两种总用时不是一个孤立架构速度对照，也不是GMT系统FPS。
CUDA分配器峰值分别4982714368/5904797184字节；整卡生命周期峰值NOT_MEASURED。
手动CUDA兼容性探针的完整外层时间未捕获，不把上述两进程合计冒充历史全部GPU账单。

独立背景重算、打包和出版核查均使用CPU，没有新模型前向、GT匹配或跟踪。
原始统计保留；最终比较采用去重端点与相同空间棋盘holdout。代码入口为
`omega_background_audit.py`、`omega_package_evidence.py`、`omega_finalize.py --diagnostic-only`、
`omega_check_publication.py`。现有输出不覆盖，不用报告器重启模型实验。

## 可复核范围与继续条件

`results/omega_numeric_evidence_v1.json`包含全部24对的原始相机/F/shape/映射、
深度与confidence分位数、实际源manifest/计时receipt和小规模对应点。
Ω的E/K有保存；旧对照只保留推导F、shape和映射，没有保存其瞬态E/K，
不从F伪造旧绝对外参，也不为补历史记录重跑已经完成的调用。

所有主语义是“已认证原生主关联wrong/missed”，并不是对最终GT身份数值的判定。
原生适用性统计不是Ω纠错；背景gate通过也不是目标身份成功。
10项坐标/权重/精度契约及3项独立端点防伪支持测试通过；出版核查复算96项
model/order背景指标并核验原工作树/固定源未变。

合法获得416 reproduction原始PT后，核验SHA
`e900833330eb0e570e10543e6108d0dd704fa7c8dbc32738a645987875298896`，
使用相同冻结首24组和独立新tag先验证416接口。不要重复512/旧VGGT模型调用。
已有小对应点可以复用；若416有效裁剪区域不同，在共同有效点上重算统计即可。
只有主几何与真实身份净决策gate成立后才开发/接入在线残差并执行完整视频。

# WWW research progress snapshot

更新时间：2026-10-05 01:14 UTC

这份文件只记录代码版本、实验阶段和本地运行状态。数据集、模型 checkpoint、推理 trace 以及机密配置不纳入 Git 提交。

## 代码状态

- 主仓库：`main`，当前本地 HEAD 为 `370440c`（研究代码基线为 `e3c2e06`）；本地相对远端 `main` 含研究代码提交，远端 `main` 保持不改写。
- JEV 隔离实现：`/data1/liuyeqiang/WWW_jev_v2`，分支 `jev/reviewer-proof-v2`，最新本地提交为 `ba63b71`；除旧版固定 22 序列名表的 MOT side writer 保护外，正式反事实 trace 现在会在解析阶段按 shard 的 `video_id` 过滤，不再把全量 trace 物化到内存。
- 两个分支均指向 GitHub 仓库 `LYQ1107/WWW`；主线远端存在未合并提交，因此推送时使用独立的研究进度分支，不改写远端 `main`。

## 训练状态

- GMT Stage1：已完成并通过 checkpoint 验证，使用 `model_16000.pth`。
- GMT Stage2：单卡已完成 `20000` iter，最终 checkpoint `/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth` 已通过 reload、finiteness、optimizer-state 验证；SHA256 为 `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`。
- 当前 Stage2 继续使用单 GPU 方案；不会因多卡同步而切换训练路径。

## Model 4500 研究流水线

- 最终 v2 的 OFF train/test/native_test 已完成，三路均使用已验证的 Stage2 `model_20000.pth`；test/native 原始任务分别在 GPU3/GPU6 完成，train 在 GPU2 的首轮任务因旧 MOT side writer 越界中断后完成可恢复尾段。
- VISION_train 含 24 个视频组、VISION_test 含 22 个视频组。train 的旧流覆盖图像 1–53002，尾段重跑视频 23/24 覆盖 53003–57508；合并后的 prediction stream 为 57508 行、COCO detection 为 544904 个。
- 合并后的 train cache 校验为 PASS（57508/57508 records、0 missing/duplicate/payload error），trace/cache 对齐校验为 PASS（1,112,173 events、0 malformed、0 missing cache key）；失败源保留在 `.failed_partial_20261005` 备份路径，未删除。
- 官方 JEV 在线测试已完成；正式 GMT 反事实分片在 OFF 阶段通过后启动。首次四 shard 并发尝试因每个进程都把完整 12.3 GiB train trace 物化到内存而触发内核 OOM，失败日志和无 manifest 的状态均保留；随后先改为串行，再加入解析阶段的 shard 过滤。
- 首次完整 train OFF 推理在旧 evaluator 汇总阶段以 `IndexError` 退出（固定 22 序列名表访问 video 23）；隔离 v2 提交 `492ccd6` 捕获该 legacy side-writer 异常并继续正式 evaluator。旧失败 manifest、partial trace/cache/stream 均保留，合并结果已通过独立覆盖检查。
- 总编排器已自动跳过已完成 OFF 任务并重新进入正式反事实分片。最新观测中 shard0 在 GPU4 上运行，已读完约 12.3 GiB trace 并进入 GMT 计算，进程 RSS 约 4.5 GiB、主机内存约 11 GiB/124 GiB，尚未生成 shard0 manifest；在这些阶段全部 PASS 前不宣称最终论文数值。
- Model 4500 的 train/test 感知缓存已完成并通过校验，记录数分别为 `57508` 和 `58038`。
- 以上 Model 4500 结果仍属于运行中的 screening/proxy 阶段；在完整 OFF 推理、对齐和正式评测结束前，不把它们写成最终论文数值。

## 最新代码验证

- 使用 GMT 环境并显式加入 `third_party/CenterNet2` 后，JEV/GMT CPU 合约、counterfactual、runtime、training smoke、policy replay 和集成测试全部通过；本轮 20 个测试脚本的 `__main__` 入口均返回 PASS（环境未安装 pytest，因此未伪造 pytest 结果）。
- 最终 v2 编排器实际调用的 17 个 CLI 均通过 `--help` 解析，主线 watcher 与隔离编排器在 GMT Python 3.10 下重新编译通过。
- 流式 VisionTrack evaluator 合约测试通过：类别映射、JSON 结果落盘和空内存汇总路径均 PASS；该改动不改变模型前向或 JEV trace。
- 曾发现的 counterfactual smoke-test 失败是旧分支 runner 返回不一致 action weight；已统一为 decision-level weight，并重新通过针对性测试和完整测试套件。
- 最终链路入口已审计：`run_final_v2_pipeline.py` 只接受 canonical `model_20000.pth`，并要求正式 GMT adapter 的四个 shard 全部 PASS；当前正式 v2 编排器已经越过 checkpoint gate、OFF gate，并在运行 formal counterfactual 阶段。

## 资源与安全

- checkpoint、数据集和大体积 trace 保留在本机 `/data1/liuyeqiang`，不上传 GitHub；当前 `/data1` 可用空间约 `48 GiB`，运行中的/正式依据文件不清理。
- 为保证正式阶段空间，已清理不再作为启动或恢复依据的 31 个旧 Stage2 checkpoint，释放约 18.4 GiB；保留代理所需的 `model_4500.pth`、Stage2 启动依据 `model_16000.pth`、恢复点 `model_17000.pth`、已验证的 `model_17500.pth` 和 `model_18000.pth`，验证 JSON 和训练日志未删除。当前 `/data1` 可用空间约 `48 GiB`，正在写入的 trace 不清理。
- 旧的未完成 proxy trace 已移动到 `/home/liuyeqiang/old_proxy_trace_quarantine/`，可恢复，未删除。
- 本次提交不包含 `results/`、模型文件、日志大文件、百度云文件或任何登录凭据。

## 下一步

1. 完成正式 GMT 反事实 train/test 分片及合并校验。
2. 完成策略训练、校准、压力测试、selection lock、oracle/反事实审计和正式评测。
3. 核对最终指标、复现实验条件、失败重跑记录并汇总研究报告。

## GitHub 同步状态

- 代码和本进度文档同步到 GitHub 独立研究分支；生成的 checkpoint、数据集、trace、结果目录和凭据不会纳入提交。
- 主研究分支：`codex/research-progress-20261004-api`（已推送研究快照）；JEV v2 基线分支：`codex/jev-reviewer-proof-v2-api-20261004`；本轮修复分支：`codex/jev-writer-guard-20261004`，远端 commit `d5c7170`。远端 `main` 保持不改写。
- 反事实 shard 串行调度分支：`codex/sequential-formal-shards-20261005`，远端 commit `399abc5`；本轮流式过滤修复分支：`codex/filtered-counterfactual-shards-20261005`，远端 commit `c505714`。

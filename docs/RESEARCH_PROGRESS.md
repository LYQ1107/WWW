# WWW research progress snapshot

更新时间：2026-10-04 20:39 UTC

这份文件只记录代码版本、实验阶段和本地运行状态。数据集、模型 checkpoint、推理 trace 以及机密配置不纳入 Git 提交。

## 代码状态

- 主仓库：`main`，当前本地 HEAD 为 `e3c2e06`；本地相对远端 `main` 含研究代码提交，远端 `main` 保持不改写。
- JEV 隔离实现：`/data1/liuyeqiang/WWW_jev_v2`，分支 `jev/reviewer-proof-v2`，最新本地提交为 `045886a`，包含最终在线 JEV 审计 trace 的 gzip 支持。
- 两个分支均指向 GitHub 仓库 `LYQ1107/WWW`；主线远端存在未合并提交，因此推送时使用独立的研究进度分支，不改写远端 `main`。

## 训练状态

- GMT Stage1：已完成并通过 checkpoint 验证，使用 `model_16000.pth`。
- GMT Stage2：单卡已完成 `20000` iter，最终 checkpoint `/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth` 已通过 reload、finiteness、optimizer-state 验证；SHA256 为 `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`。
- 当前 Stage2 继续使用单 GPU 方案；不会因多卡同步而切换训练路径。

## Model 4500 研究流水线

- 最终 v2 流水线已启动，当前阶段为 `final_off_inference`；train/test/native_test 分别占用 GPU2/GPU3/GPU6，使用已验证的 Stage2 `model_20000.pth`。
- VISION_train 含 24 个视频组、VISION_test 含 22 个视频组；三路任务正在逐视频组完成 OFF 感知缓存、JEV trace 和流式预测写出，当前仍属运行中，未宣称最终论文数值。
- 官方 JEV 在线测试、反事实分片、策略训练和正式评测将在 OFF 阶段通过后由同一编排器继续；JEV 长轨迹溢出修复已进入隔离分支。
- 首次完整 train OFF 推理已写完约 `968195` 个决策事件，但在旧的 COCO evaluator 汇总阶段退出，未形成结果文件；失败输出和 partial trace 已保留为带时间戳的 `.failed_*` 证据。隔离 v2 已提交 `8d7078c`，对 `VISION_train/test` 改为流式写结果；随后由 `fc65925` 恢复 evaluator 调用、`a20612b` 补回 `torch.no_grad()`，当前 GPU2/GPU3 重跑正常推进。
- 正式流水线 watcher 当前等待完整 OFF 推理结束，之后会自动生成并校验正式 GMT 对齐数据，再启动正式评测。
- Model 4500 的 train/test 感知缓存已完成并通过校验，记录数分别为 `57508` 和 `58038`。
- 以上 Model 4500 结果仍属于运行中的 screening/proxy 阶段；在完整 OFF 推理、对齐和正式评测结束前，不把它们写成最终论文数值。

## 最新代码验证

- 使用 GMT 环境并显式加入 `third_party/CenterNet2` 后，JEV/GMT CPU 合约、counterfactual、runtime、training smoke、policy replay 和集成测试全部通过；本轮 20 个测试脚本的 `__main__` 入口均返回 PASS（环境未安装 pytest，因此未伪造 pytest 结果）。
- 最终 v2 编排器实际调用的 17 个 CLI 均通过 `--help` 解析，主线 watcher 与隔离编排器在 GMT Python 3.10 下重新编译通过。
- 流式 VisionTrack evaluator 合约测试通过：类别映射、JSON 结果落盘和空内存汇总路径均 PASS；该改动不改变模型前向或 JEV trace。
- 曾发现的 counterfactual smoke-test 失败是旧分支 runner 返回不一致 action weight；已统一为 decision-level weight，并重新通过针对性测试和完整测试套件。
- 最终链路入口已审计：`run_final_v2_pipeline.py` 只接受 canonical `model_20000.pth`，并要求正式 GMT adapter 的四个 shard 全部 PASS；当前正式 v2 编排器已经越过 checkpoint gate 并在运行 OFF 阶段。

## 资源与安全

- checkpoint、数据集和大体积 trace 保留在本机 `/data1/liuyeqiang`，不上传 GitHub。
- 为保证正式阶段空间，已清理不再作为启动或恢复依据的 31 个旧 Stage2 checkpoint，释放约 18.4 GiB；保留代理所需的 `model_4500.pth`、Stage2 启动依据 `model_16000.pth`、恢复点 `model_17000.pth`、已验证的 `model_17500.pth` 和 `model_18000.pth`，验证 JSON 和训练日志未删除。当前 `/data1` 可用空间约 `96 GiB`，正在写入的 trace 不清理。
- 旧的未完成 proxy trace 已移动到 `/home/liuyeqiang/old_proxy_trace_quarantine/`，可恢复，未删除。
- 本次提交不包含 `results/`、模型文件、日志大文件、百度云文件或任何登录凭据。

## 下一步

1. 等待 OFF train/test/native_test 完成，自动生成正式对齐数据和 GMT/JEV 评测。
2. 核对最终指标、策略选择门禁和 oracle/反事实审计结果。
3. 汇总可复现实验条件、最终指标、失败重跑记录和研究报告。

## GitHub 同步状态

- 代码和本进度文档已同步到 GitHub 独立研究分支；生成的 checkpoint、数据集、trace、结果目录和凭据不会纳入提交。
- 主研究分支：`codex/research-progress-20261004-api`，远端 commit `9b5d18c`；JEV v2 分支：`codex/jev-reviewer-proof-v2-api-20261004`，远端 commit `87065dc`。远端 `main` 保持不改写。

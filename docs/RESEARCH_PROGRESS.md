# WWW research progress snapshot

更新时间：2026-10-04 17:34 UTC

这份文件只记录代码版本、实验阶段和本地运行状态。数据集、模型 checkpoint、推理 trace 以及机密配置不纳入 Git 提交。

## 代码状态

- 主仓库：`main`，当前本地 HEAD 为 `9256072`；本地相对 `origin/main` 含研究代码提交，远端仍有 3 个未合并提交。
- JEV 隔离实现：`/data1/liuyeqiang/WWW_jev_v2`，分支 `jev/reviewer-proof-v2`，最终 v2 编排器提交为 `ec8af4e`（建立在 `740dd85` 的 counterfactual decision-weight 一致性修复之上）。
- 两个分支均指向 GitHub 仓库 `LYQ1107/WWW`；主线远端存在未合并提交，因此推送时使用独立的研究进度分支，不改写远端 `main`。

## 训练状态

- GMT Stage1：已完成并通过 checkpoint 验证，使用 `model_16000.pth`。
- GMT Stage2：单卡持续运行，目标 `20000` iter；已保存 `model_18500.pth`，最近观测到 `18540/20000`，loss 仍有限、显存约 `10221 MB`。`model_18000.pth` 验证 PASS（iteration/global=18000、reload、finiteness、optimizer 均 PASS，SHA256 为 `4f93f8e19d172bb7f46631c05927627f269c4e7a9c83db44cecb524b425b1971`）；最终 checkpoint 等待器仍在等待 `model_20000.pth`。
- 当前 Stage2 继续使用单 GPU 方案；不会因多卡同步而切换训练路径。

## Model 4500 研究流水线

- train/test 的完整 OFF 感知推理仍在运行（GPU2/GPU3），分别写入 `train_off_full_v2.jsonl` 和 `test_off_full_v2.jsonl`；17:32 UTC 时 trace 约为 283 MB/483 MB，两个流式 COCO 结果文件尚未生成，进程内存稳定且未再次 OOM。
- 官方 JEV 在线测试正在运行，使用 `jev_h64` 控制器和独立 trace；JEV 长轨迹溢出修复已经进入隔离分支。
- 官方 OFF 测试已遍历到 `1052/1052`，进程正在完成结果写盘；完整 train/test OFF trace 和 JEV trace 仍在运行。
- native/off 官方 baseline 仍在运行或完成收尾。
- 首次完整 train OFF 推理已写完约 `968195` 个决策事件，但在旧的 COCO evaluator 汇总阶段退出，未形成结果文件；失败输出和 partial trace 已保留为带时间戳的 `.failed_*` 证据。隔离 v2 已提交 `8d7078c`，对 `VISION_train/test` 改为流式写结果；随后由 `fc65925` 恢复 evaluator 调用、`a20612b` 补回 `torch.no_grad()`，当前 GPU2/GPU3 重跑正常推进。
- 正式流水线 watcher 当前等待完整 OFF 推理结束，之后会自动生成并校验正式 GMT 对齐数据，再启动正式评测。
- Model 4500 的 train/test 感知缓存已完成并通过校验，记录数分别为 `57508` 和 `58038`。
- 以上 Model 4500 结果仍属于运行中的 screening/proxy 阶段；在完整 OFF 推理、对齐和正式评测结束前，不把它们写成最终论文数值。

## 最新代码验证

- 使用 GMT 环境并显式加入 `third_party/CenterNet2` 后，JEV/GMT CPU 合约、counterfactual、runtime、training smoke、policy replay 和集成测试全部通过；本轮 20 个测试脚本的 `__main__` 入口均返回 PASS（环境未安装 pytest，因此未伪造 pytest 结果）。
- 最终 v2 编排器实际调用的 17 个 CLI 均通过 `--help` 解析，主线 watcher 与隔离编排器在 GMT Python 3.10 下重新编译通过。
- 流式 VisionTrack evaluator 合约测试通过：类别映射、JSON 结果落盘和空内存汇总路径均 PASS；该改动不改变模型前向或 JEV trace。
- 曾发现的 counterfactual smoke-test 失败是旧分支 runner 返回不一致 action weight；已统一为 decision-level weight，并重新通过针对性测试和完整测试套件。
- 最终链路入口已审计：`chain_research_after_stage2_v2.py` 会等待当前 Model 4500 的 OFF 推理及 formal finisher 释放 GPU 后，调用隔离 worktree 中的 `run_final_v2_pipeline.py`；该编排器只接受 canonical `model_20000.pth`，并要求正式 GMT adapter 的四个 shard 全部 PASS。最终链路当前尚未启动，因为 `model_20000.pth` 尚未生成；`outputs/research_v2/ARM_FINAL_PIPELINE` 已 arm，但仍受 checkpoint validation gate 约束。

## 资源与安全

- checkpoint、数据集和大体积 trace 保留在本机 `/data1/liuyeqiang`，不上传 GitHub。
- 为保证正式阶段空间，已清理不再作为启动或恢复依据的 31 个旧 Stage2 checkpoint，释放约 18.4 GiB；保留代理所需的 `model_4500.pth`、Stage2 启动依据 `model_16000.pth`、恢复点 `model_17000.pth`、已验证的 `model_17500.pth` 和 `model_18000.pth`，验证 JSON 和训练日志未删除。当前 `/data1` 可用空间约 `96 GiB`，正在写入的 trace 不清理。
- 旧的未完成 proxy trace 已移动到 `/home/liuyeqiang/old_proxy_trace_quarantine/`，可恢复，未删除。
- 本次提交不包含 `results/`、模型文件、日志大文件、百度云文件或任何登录凭据。

## 下一步

1. 等待 Stage2 `model_20000.pth` 并完成校验。
2. 等待 OFF train/test trace 完成，自动生成正式对齐数据和 GMT/JEV 评测。
3. 汇总可复现实验条件、最终指标、失败重跑记录和研究报告。

## GitHub 同步状态

- 代码和本进度文档会提交到本地分支历史；生成的 checkpoint、数据集、trace、结果目录和凭据不会纳入提交。
- 由于当前 remote 使用 HTTPS 且本机没有可用 GitHub 凭据，推送需在凭据可用后执行；本地提交哈希会在交接时列出。
- 本次实际尝试的推送引用为 `codex/research-progress-20261004`（主仓库）和 `codex/jev-reviewer-proof-v2-20261004`（隔离分支），均因 `could not read Username for 'https://github.com'` 返回失败；远端没有声称已更新。

# WWW / JEV 项目存储清理：2026-10-12

已完成 home 与 data1 的项目存储清理，按文件实际占用块统计共释放 **43.436 GiB**。六份旧 TRAIN JSONL 日志无损归档，原始失败和负面证据仍保留在压缩包内；另清理可重新下载的安装缓存。

| 文件系统 | 本次可归因释放（GiB） | 完成时可用（GiB） |
| --- | ---: | ---: |
| home（根分区） | 16.100 | 53.090 |
| data1 | 27.336 | 53.421 |

`df` 可用空间会随其他进程读写变化，不能把整个分区的变化都算作本次释放。JSON 同时保存开始盘点、执行前、执行后三个快照；释放量采用原文件占用减归档占用，缓存仅计被删除且无其他硬链接的文件块，未计小量目录块。

## 归档记录

| 原文件 | 原占用（GiB） | 保留归档（GiB） | 释放（GiB） |
| --- | ---: | ---: | ---: |
| `/home/liuyeqiang/old_proxy_trace_quarantine/train_off_gpu2_isolated.jsonl` | 7.608 | 0.357 | 7.251 |
| `/home/liuyeqiang/WWW_failed_artifacts/train_off.failed_partial_20261005.jsonl` | 8.817 | 0.424 | 8.393 |
| `/data1/liuyeqiang/WWW/outputs/research_v2/model_4500/traces/train_off_full_v2.failed_20261004T165344Z.jsonl` | 7.547 | 0.392 | 7.155 |
| `/data1/liuyeqiang/WWW/outputs/research_v2/model_4500/traces/train_off_full_v2.jsonl` | 4.153 | 0.239 | 3.914 |
| `/data1/liuyeqiang/WWW/outputs/research_final_v2/off/traces/train_off.jsonl` | 11.455 | 0.527 | 10.927 |
| `/data1/liuyeqiang/WWW/outputs/research_final_v2/off/traces/train_tail_off.jsonl` | 5.573 | 0.234 | 5.340 |

每个归档位于原路径加 `.zst`。使用现有 Zstd 1.4.9、level 1、两条 CPU 工作线程，未使用 GPU。流式计算原文件 SHA256，再完整解压计算 SHA256 和字节数；结果完全相同后检查 inode、大小、mtime、ctime 未变，再移除原始未压缩副本。删除后的六个归档也已再次确认存在且压缩包 SHA256 匹配。完整原始/压缩 SHA 和逐文件时间见 [清理记录](../reports/JEV_MASTER/STORAGE_CLEANUP_20261012.json)。

可读取的同用户进程 fd、映射和精确路径参数均经过占用检查。SSH/SFTP 服务因 Linux 权限无法检查 fd/maps，该限制已逐项记录，未声称检查覆盖所有打开的文件。原日志均超过 24 小时未修改，压缩与校验全过程要求源文件元数据保持不变。

## 缓存与保留范围

Conda 仅清 tarballs 和 index-cache，pip 仅清 HTTP/selfcheck 缓存，释放 0.456 GiB。没有删除 Conda 环境、已安装包目录、HF 模型、MATLAB 安装目录、原始视频/标注、checkpoint、当前 perception/native 缓存、官方 TEST 或 sealed 20/21/22。其他项目和其他用户文件未处理。重复 checkpoint 的预检显示重复路径已指向同一 inode，未再重复计算或删除“副本”。

Phase XVII–XXI 的 203 个已发布代码/报告文件 SHA256 全部匹配；核心 GMT、两份 Qwen 权重分片、QA adapter 共四个权重文件 SHA256 也全部匹配。未宣称重新校验所有历史权重。未重跑科学实验，MASTER GOAL 的证据门槛与状态保持不变。

## 还原方法

旧程序按原始 `.jsonl` 路径读取时，需先还原对应文件。工具不会将压缩内容伪装成原始 JSONL。

```bash
/home/liuyeqiang/anaconda3/envs/GMT/bin/python \
  /home/liuyeqiang/WWW_jev_master_audit/reproduction_tools/jev_master_storage_cleanup.py \
  --restore /完整/原始/train日志.jsonl
```

`--restore` 后应填写上表中的一个完整原路径。默认读取本机 `/home/liuyeqiang/WWW_jev_master_audit_runtime/storage_cleanup_20261012/ARCHIVE_LEDGER.json`；也可用 `--root` 指定包含同名台账的目录。还原会验证压缩包 SHA、解压 SHA 和字节数，拒绝覆盖已有文件，并要求原文件大小加 5 GiB 的剩余空间。压缩包继续保留。

自测覆盖完整还原、拒绝覆盖、已有归档、错误 SHA、源文件变动、活跃进程占用。六份实际归档均通过完整解压校验。GitHub 只上传代码、报告、紧凑 JSON 与 SHA 清单，归档包留在服务器。

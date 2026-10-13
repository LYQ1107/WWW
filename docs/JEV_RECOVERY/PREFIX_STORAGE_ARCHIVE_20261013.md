# 空间保留线恢复：2026-10-13

原R2已完成机制Gate；R3 Tiny/Pilot和原始Fixed完整TRAIN16已完成。随后/data1剩余空间低于30GiB，发布保护导致监督/刷新/等待式finisher停止。没有把保护阈值降低，也没有重训已完成权重。

本轮归档221个旧Phase X TRAIN prefix状态缓存（video12/13/14/16/17/18/19），完整解压后逐成员SHA和字节数一致，再逐项检查源元数据和可读取同用户进程引用，释放25.361GiB实际文件块。原25.763GiB缓存由约0.401GiB solid zstd/tar保留。当前R2/R3使用的是Phase XV密集输入及prefix，这些原Phase X缓存不是当前训练输入。模型checkpoint、原指标报告、perception/native当前缓存、TEST和sealed20/21/22未归档。

归档位于/data1/liuyeqiang/WWW_storage_maintenance/20261013_prefix_archive/TRAIN_PREFIXES.tar.zst；逐成员原路径/SHA/大小记录于同目录ARCHIVE_LEDGER.json，紧凑副本见reports/JEV_RECOVERY/PREFIX_STORAGE_ARCHIVE_20261013.json。可读取同用户进程fd/maps沿用既有工具检查；SSH/SFTP权限限制未声称已覆盖。

旧prefix原始路径需要先恢复再被torch读取，没有把压缩容器伪装成原始PTH。恢复命令：

```bash
/home/liuyeqiang/anaconda3/envs/GMT/bin/python /home/liuyeqiang/WWW_jev_recovery_20261012/reproduction_tools/jev_recovery_archive_prefixes.py --restore-member video13/compat/prefixes/PREFIX_13_000002_1.pth
```

恢复拒绝覆盖已有文件，核验归档SHA、成员原SHA/长度，再恢复原路径权限与时间戳。已经验证一个实际成员完整还原。归档及恢复不改变旧模型/科学结果；恢复时应留成员大小加5GiB空间。

监督器恢复时复用COMPLETE登记项，R3预注册和已冻结Gate须逐字段相同，拒绝用重启修改原设计。Git只上传代码、记录、SHA，不上传归档包。

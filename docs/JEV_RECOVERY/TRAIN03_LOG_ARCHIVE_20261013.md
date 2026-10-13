# 停用TRAIN03大日志无损归档

两份旧Phase VI TRAIN03日志完整压缩/解压SHA和字节数一致，源元数据未变且可检查同用户进程没有引用后才释放raw副本。本轮可归因释放1.288GiB；模型权重、指标结果、当前TRAIN输入和原Gate未改。压缩包保存在原路径加.zst，完整记录见TRAIN03_LOG_ARCHIVE_20261013.json。

需要旧原始路径时可恢复，禁止将压缩内容伪装成原始JSON/JSONL。恢复命令示例：

```bash
/home/liuyeqiang/anaconda3/envs/GMT/bin/python /home/liuyeqiang/WWW_jev_master_audit/reproduction_tools/jev_master_storage_cleanup.py --root /data1/liuyeqiang/WWW_storage_maintenance/20261013_train03_logs --restore /data1/liuyeqiang/WWW_storage_maintenance/20261010/home_runtime/WWW_jev_phase6_runtime/20261008/heldouts/video03/G2/online_decisions.jsonl
```

归档SHA和raw SHA/大小逐项核验；既有恢复工具拒绝覆盖文件。SSH/SFTP进程fd/maps权限限制逐项列在记录中，不声称完全可检查所有服务。未读取TEST或sealed20/21/22，也没有把分区整体空间变化全归因于本次清理。

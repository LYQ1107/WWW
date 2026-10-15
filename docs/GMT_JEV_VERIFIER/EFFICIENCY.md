# 效率证据边界

冻结历史同GPU7完整图像FPS：GMT3.409768；MLP3.129765；缓存MLP2.848982。默认和缓存均失败95%门槛。这是V1测量，本轮未产生新的完整FPS。
P0每个包有额外诊断求解与GT标注，不用其帧率冒充部署速度。当前只训练/诊断Q1899参数，它未接入部署；新Typed开销、参数与FPS均NOT_RUN。
原MLP24帧GPU组件诊断显示feature p50约9.16ms、head约.537ms，仅为前缀定位；不外推全视频，不因此声称缓存更快。
只有新模型真实完整在线质量合格后才启动新同卡完整图像效率测量；旧九次交错方案明确未执行。
资源账本：{"created_Asia_Shanghai": "2026-10-15T15:17:07.653476+08:00", "new_GPU_hours_including_live": 1.1772130608724223, "new_GPU_hours_max": 4, "old_GPU_hours_unchanged": 11.92904806540865, "new_retained_MiB": 1.4122848510742188, "home_free_GiB": 50.00343704223633, "registered_live_jobs": []}

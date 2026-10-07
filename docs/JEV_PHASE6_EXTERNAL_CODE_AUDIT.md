# JEV Phase VI 外部源码审计

审计日期：2026-10-08（Asia/Shanghai）。在修改本项目跟踪/训练代码之前，逐行检查以下 pinned source 的指定函数实现与实际调用路径。这里记录源码事实、对本项目的推论及移植限制；不把 README 的方法描述当作实现证据。下载快照仅用于阅读，未执行外部代码或安装其依赖。

20 个 repository 的完整 commit SHA、下载文件 SHA256 与函数锚点另见 `reports/JEV_PHASE6/EXTERNAL_SOURCE_MANIFEST.json`。源码事实均链接到冻结 commit，研究启发为本项目推论。

## 01. FoundationVision/ByteTrack

Commit：`d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`。

- [yolox/tracker/byte_tracker.py](https://github.com/FoundationVision/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/byte_tracker.py)：`STrack.re_activate`, `BYTETracker.update`.

**源码事实：** 高置信检测与 tracked+lost pool 先关联，低置信检测只与剩余 tracked 关联；未确认、新建、超时移除分开处理。

**JEV 推论：** MATCH/REACT 的 state transition 必须独立；REASSOCIATE 不等于拒绝后直接新建。

**不能照搬：** 不能复制 MOT17 阈值或把低置信第二轮称为完整 stale-bank 恢复；当前主要是 IoU/Kalman。

## 02. NirAharon/BoT-SORT

Commit：`251985436d6712aaf682aaaf5f71edb4987224bd`。

- [tracker/bot_sort.py](https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/bot_sort.py)：`STrack.update_features`, `STrack.re_activate`, `BoTSORT.update`.

**源码事实：** 固定 EMA 后归一化；appearance 由 proximity/appearance 阈值筛选，lost/refind 与两轮关联显式区分。

**JEV 推论：** M3 应严格使用 old-weight alpha；motion/appearance 约束与 memory 表示一起审计。

**不能照搬：** 源码的固定 EMA 本身不是 detector-confidence 自适应，也不证明 learned memory gate 有价值；不能直接复用其 CMC/ReID 模型。

## 03. dyhBUPT/StrongSORT

Commit：`ee995076da5083e28d0da1f885297df62705ebd7`。

- [deep_sort/tracker.py](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/deep_sort/tracker.py)：`Tracker.update`, `Tracker._match`.
- [deep_sort/track.py](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/deep_sort/track.py)：`Track.update`.
- [deep_sort/nn_matching.py](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/deep_sort/nn_matching.py)：`NearestNeighborDistanceMetric.partial_fit`.
- [strong_sort.py](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/strong_sort.py)：`module main`.
- [AFLink/AppFreeLink.py](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/AFLink/AppFreeLink.py)：`AFLink.link`, `AFLink.predict`, `AFLink.compression`.
- [AFLink/model.py](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/AFLink/model.py)：`PostLinker.forward`.

**源码事实：** confirmed appearance cascade + IOU fallback；可选 EMA；AFLink 在整段输出后按时间/空间门控与 learned link cost 做轨迹 Hungarian 合并。

**JEV 推论：** 短期 MATCH 与长期恢复可采用不同输入和时间尺度；controlled augmentation 应来自实际状态。

**不能照搬：** AFLink 使用完整已结束轨迹，是离线后处理，不能照搬为因果在线 REACT；其神经网络和结果重写不能当作 JEV 的原生等价实现。

## 04. nwojke/deep_sort

Commit：`f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab`。

- [deep_sort/tracker.py](https://github.com/nwojke/deep_sort/blob/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/tracker.py)：`Tracker.update`, `Tracker._match`.
- [deep_sort/nn_matching.py](https://github.com/nwojke/deep_sort/blob/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/nn_matching.py)：`NearestNeighborDistanceMetric.partial_fit`, `NearestNeighborDistanceMetric.distance`.
- [deep_sort/track.py](https://github.com/nwojke/deep_sort/blob/f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab/deep_sort/track.py)：`Track.update`, `Track.mark_missed`.

**源码事实：** 只有 confirmed tracks 的暂存 features 进入 metric partial_fit；按 budget 留最近 gallery，并删除非 active targets；appearance cascade 后对未确认/最近失配做 IOU fallback。

**JEV 推论：** maturity、bounded gallery 与匹配确认条件应进入 MEMORY 诊断；M2 的距离聚合须明确。

**不能照搬：** DeepSORT 最近邻 gallery 与 Tracktor mean gallery 不相同；删除 gallery 的语义也不能直接当长期 stale-bank 保留。

## 05. noahcao/OC_SORT

Commit：`8462e7e729a93ccd3bd995c0a79a890336cb3a0b`。

- [trackers/ocsort_tracker/ocsort.py](https://github.com/noahcao/OC_SORT/blob/8462e7e729a93ccd3bd995c0a79a890336cb3a0b/trackers/ocsort_tracker/ocsort.py)：`KalmanBoxTracker.update`, `OCSort.update`.
- [trackers/ocsort_tracker/association.py](https://github.com/noahcao/OC_SORT/blob/8462e7e729a93ccd3bd995c0a79a890336cb3a0b/trackers/ocsort_tracker/association.py)：`associate`, `linear_assignment`.

**源码事实：** 第一轮 IoU+观测方向代价；可选低分 BYTE；随后剩余检测与 last_observation 框重新 assignment，再处理 missing/new/dead。

**JEV 推论：** REASSOCIATE 是独立操作，必须记录候选变化与真正 assigned ID。

**不能照搬：** 这里 OCR 改用了 observation bbox；JEV 的重新指派禁止当前 edge 并复用同一矩阵，二者不是同一个算子。

## 06. GerardMaggiolino/Deep-OC-SORT

Commit：`6bb51d027b137233f5c520b6fcc4f2ae387a6ba9`。

- [trackers/integrated_ocsort_embedding/ocsort.py](https://github.com/GerardMaggiolino/Deep-OC-SORT/blob/6bb51d027b137233f5c520b6fcc4f2ae387a6ba9/trackers/integrated_ocsort_embedding/ocsort.py)：`KalmanBoxTracker.update_emb`, `OCSort.update`.
- [trackers/integrated_ocsort_embedding/association.py](https://github.com/GerardMaggiolino/Deep-OC-SORT/blob/6bb51d027b137233f5c520b6fcc4f2ae387a6ba9/trackers/integrated_ocsort_embedding/association.py)：`compute_aw_new_metric`, `associate`.

**源码事实：** trust=(confidence-det_thresh)/(1-det_thresh)；alpha 随低置信趋近1，EMA 更新趋近0；appearance 权重同时利用行/列 top1-top2 差异；OCR 第二轮仍可基于 IoU。

**JEV 推论：** M4 使用 confidence-adaptive EMA；MATCH 需要证明 ambiguity/candidate interaction 提供额外闭环收益。

**不能照搬：** confidence 不是 GT 正确率；上游 det_thresh 和 embedding normalisation 不能脱离当前冻结感知契约照搬。

## 07. ymzis69/HybridSORT

Commit：`396f8d30db13304c0cbaf1dcf2e16ded93ce1701`。

- [trackers/byte_tracker/byte_tracker.py](https://github.com/ymzis69/HybridSORT/blob/396f8d30db13304c0cbaf1dcf2e16ded93ce1701/trackers/byte_tracker/byte_tracker.py)：`BYTETracker.update`.
- [trackers/ocsort_tracker/ocsort.py](https://github.com/ymzis69/HybridSORT/blob/396f8d30db13304c0cbaf1dcf2e16ded93ce1701/trackers/ocsort_tracker/ocsort.py)：`OCSort.update`.
- [utils/args.py](https://github.com/ymzis69/HybridSORT/blob/396f8d30db13304c0cbaf1dcf2e16ded93ce1701/utils/args.py)：`make_parser`.
- [trackers/hybrid_sort_tracker/hybrid_sort.py](https://github.com/ymzis69/HybridSORT/blob/396f8d30db13304c0cbaf1dcf2e16ded93ce1701/trackers/hybrid_sort_tracker/hybrid_sort.py)：`Hybrid_Sort.update`.

**源码事实：** 指定 ByteTrack/OCSort 文件是基础实现；真正 TCM 开关调用在 hybrid_sort.py，第一轮融合四角速度/score，BYTE 阶段显式减去 score-difference cost；args 区分高/低分匹配阈值与 longterm bank。

**JEV 推论：** weak cues 与各阶段门控需单独验证；G1/G2 不应偷用三动作约束或二次 assignment。

**不能照搬：** 不能把基础 ByteTrack 文件误报为全部 HybridSORT 算法；不同 confidence score 和几何 cues 的尺度需重新绑定。

## 08. ifzhang/FairMOT

Commit：`4aa62976bde6266cbafd0509e24c3d98a7d0899f`。

- [src/lib/tracker/multitracker.py](https://github.com/ifzhang/FairMOT/blob/4aa62976bde6266cbafd0509e24c3d98a7d0899f/src/lib/tracker/multitracker.py)：`STrack.update_features`, `STrack.re_activate`, `JDETracker.update`.

**源码事实：** appearance+Kalman motion 的第一轮含 lost pool，剩余 tracked 用 IOU；EMA appearance 与 lost buffer 明确。

**JEV 推论：** 显式 active/lost/refind/new 生命周期支持 typed state；memory update 不等于普通 association history。

**不能照搬：** FairMOT 联合训练 detector/ReID，不能把改变感知 backbone 的收益算在 JEV controller 上。

## 09. xingyizhou/CenterTrack

Commit：`e4e7534cc2ebfbd31e0cde680988f286c65fe34f`。

- [src/lib/utils/tracker.py](https://github.com/xingyizhou/CenterTrack/blob/e4e7534cc2ebfbd31e0cde680988f286c65fe34f/src/lib/utils/tracker.py)：`Tracker.step`, `Tracker.init_track`, `greedy_assignment`.

**源码事实：** 检测中心+tracking offset 指向过去，按面积/类别门控 Hungarian 或 greedy；new_thresh 与 max_age 仍独立管理出生和暂失。

**JEV 推论：** learned matching 表示与 birth decision 仍可分开；保持冻结 perception 的公平对比。

**不能照搬：** offset 来自另外训练的 detector；当前代码未匹配旧 track 的平移为零，不应声称通用 learned recovery。

## 10. SysCV/qdtrack

Commit：`c5b10472d7bdd3b9ab75255dd10e48e21f48c54f`。

- [qdtrack/models/trackers/quasi_dense_embed_tracker.py](https://github.com/SysCV/qdtrack/blob/c5b10472d7bdd3b9ab75255dd10e48e21f48c54f/qdtrack/models/trackers/quasi_dense_embed_tracker.py)：`QuasiDenseEmbedTracker.update_memo`, `QuasiDenseEmbedTracker.match`, `QuasiDenseEmbedTracker.memo`.
- [configs/mot17/qdtrack-frcnn_r50_fpn_4e_mot17.py](https://github.com/SysCV/qdtrack/blob/c5b10472d7bdd3b9ab75255dd10e48e21f48c54f/configs/mot17/qdtrack-frcnn_r50_fpn_4e_mot17.py)：`tracker config`.
- [configs/_base_/qdtrack_faster_rcnn_r50_fpn.py](https://github.com/SysCV/qdtrack/blob/c5b10472d7bdd3b9ab75255dd10e48e21f48c54f/configs/_base_/qdtrack_faster_rcnn_r50_fpn.py)：`model config`.

**源码事实：** tracklet 与 backdrop 使用不同留存期限；embed=(1-momentum)*old+momentum*new；bisoftmax 竞争和 confidence/类别筛选后更新 memo。

**JEV 推论：** M5 的 momentum 是 new-weight，和 BoT EMA old-weight 相反；retention 与 WRITE gate 必须分开比较。

**不能照搬：** 新特征权重不能直接写成旧权重；backdrop/类别过滤不能无说明加入当前单类 GMT。

## 11. phil-bergmann/tracking_wo_bnw

Commit：`446abc7fe65e21068a97fa8234ca1176f0a34303`。

- [src/tracktor/tracker.py](https://github.com/phil-bergmann/tracking_wo_bnw/blob/446abc7fe65e21068a97fa8234ca1176f0a34303/src/tracktor/tracker.py)：`Tracker.reid`, `Tracker.tracks_to_inactive`, `Tracker.step`, `Track.add_features`, `Track.test_features`.
- [experiments/cfgs/tracktor.yaml](https://github.com/phil-bergmann/tracking_wo_bnw/blob/446abc7fe65e21068a97fa8234ca1176f0a34303/experiments/cfgs/tracktor.yaml)：`tracker config`.

**源码事实：** inactive pool 保存有限 appearance history；test_features 用均值，ReID distance 经可选 IoU 门控 Hungarian 后恢复旧 ID；配置 max_features_num10/inactive_patience50。

**JEV 推论：** bounded gallery 的恢复表示、空间约束、耐心期限应写入 MEMORY/REACT 审计。

**不能照搬：** 这是检测框回归 tracker；mean gallery 与 DeepSORT 最邻近 gallery 不能混称为同一基线，阈值200属于它的距离尺度。

## 12. timmeinhardt/trackformer

Commit：`e468bf156b029869f6de1be358bc11cd1f517f3c`。

- [src/trackformer/models/detr_tracking.py](https://github.com/timmeinhardt/trackformer/blob/e468bf156b029869f6de1be358bc11cd1f517f3c/src/trackformer/models/detr_tracking.py)：`DETRTrackingBase.add_track_queries_to_targets`, `DETRTrackingBase.__init__`.
- [cfgs/train.yaml](https://github.com/timmeinhardt/trackformer/blob/e468bf156b029869f6de1be358bc11cd1f517f3c/cfgs/train.yaml)：`track query corruption config`.

**源码事实：** 训练随机减少上帧匹配 query、从未匹配检测插入 false positives，并生成对应 masks；配置 FP0.1/FN0.4。

**JEV 推论：** clean teacher states 不能代表 self-induced states；Phase VI corruption 需修改实际 tracker 状态并重算标签。

**不能照搬：** 源码具体采样包含 uniform subset 和特殊路径，不可机械当固定 Bernoulli0.4；GT 构造仅可在训练监督，runtime 特征不能复制 GT matching。

## 13. megvii-research/MOTR

Commit：`8690da3392159635ca37c31975126acf40220724`。

- [models/qim.py](https://github.com/megvii-research/MOTR/blob/8690da3392159635ca37c31975126acf40220724/models/qim.py)：`QueryInteractionModule._select_active_tracks`, `QueryInteractionModule._add_fp_tracks`, `QueryInteractionModule._update_track_embedding`.
- [models/memory_bank.py](https://github.com/megvii-research/MOTR/blob/8690da3392159635ca37c31975126acf40220724/models/memory_bank.py)：`MemoryBank.update`, `MemoryBank._forward_temporal_attn`, `MemoryBank.forward`.
- [models/motr.py](https://github.com/megvii-research/MOTR/blob/8690da3392159635ca37c31975126acf40220724/models/motr.py)：`RuntimeTrackerBase.update`, `MOTR._post_process_single_image`.

**源码事实：** 训练 QIM 丢 track/加 false query，时间 memory 在查询前先读；inference 按 score 与 save_period 保存有限 bank；低 score 持续达到 miss_tolerance 才注销 ID。

**JEV 推论：** READ/WRITE 时序、query update 与 lifecycle 需要分别记录；memory horizon 必须延伸至第一次真正读取。

**不能照搬：** 其 temporal memory 在 active 路径也参与 attention，与 GMT stale-only bank 不等价；大 attention 模型及训练 GT-active 条件不能直接复制。

## 14. megvii-research/MOTRv2

Commit：`1aac7c3beb093d4b523419a197e9531597e01798`。

- [models/motr.py](https://github.com/megvii-research/MOTRv2/blob/1aac7c3beb093d4b523419a197e9531597e01798/models/motr.py)：`RuntimeTrackerBase.update`, `RuntimeTrackerBase.__init__`.
- [models/qim.py](https://github.com/megvii-research/MOTRv2/blob/1aac7c3beb093d4b523419a197e9531597e01798/models/qim.py)：`QueryInteractionModulev2._select_active_tracks`, `QueryInteractionModulev2._update_track_embedding`.

**源码事实：** 新对象 score>=score_thresh 分配全局 ID，旧对象 score<filter_score_thresh 累积 disappear_time；超过容忍才删除；query 高置信更新。

**JEV 推论：** birth、暂时 disappearance、deletion 必须用不同状态，canonical age 应是相对时间。

**不能照搬：** RuntimeTrackerBase 删除后的旧 ID 不由这段代码恢复；不能把 miss_tolerance 叫完整 ReID/stale-bank rescue。

## 15. MCG-NJU/MeMOTR

Commit：`eb7a177b9cbcb89742ec69b2545ab3af2ea31a80`。

- [models/query_updater.py](https://github.com/MCG-NJU/MeMOTR/blob/eb7a177b9cbcb89742ec69b2545ab3af2ea31a80/models/query_updater.py)：`QueryUpdater.update_tracks_embedding`, `QueryUpdater.select_active_tracks`, `QueryUpdater.__init__`.

**源码事实：** 短期 confidence-weighted output 与 last_output 融合，long_memory attention/norm/FFN 更新 query；只有 update_threshold 以上写 long_memory EMA，训练可做 TP drop/FP insert。

**JEV 推论：** quality-controlled anchor 与 memory representation 都值得比较；typed adapters 可以轻量分开语义。

**不能照搬：** 不能把 learned confidence vector attention 直接视为 JEV binary write-gate 对照；这里跨 track 的大表示模型超出当前预算。

## 16. dvl-tum/GHOST

Commit：`755a5dacfcf4dd122a4cac73061b24e9c84f3c19`。

- [src/tracking_utils.py](https://github.com/dvl-tum/GHOST/blob/755a5dacfcf4dd122a4cac73061b24e9c84f3c19/src/tracking_utils.py)：`get_proxy`, `Track.add_detection`.
- [src/tracker.py](https://github.com/dvl-tum/GHOST/blob/755a5dacfcf4dd122a4cac73061b24e9c84f3c19/src/tracker.py)：`Tracker.proxy_dist`, `Tracker.get_hungarian_each_sample`, `Tracker.assign`, `Tracker.assign_separatly`, `Tracker.assign_act_inact_same_time`.
- [config/config_tracker.yaml](https://github.com/dvl-tum/GHOST/blob/755a5dacfcf4dd122a4cac73061b24e9c84f3c19/config/config_tracker.yaml)：`active/inactive appearance config`.
- [config/config_tracker_dance.yaml](https://github.com/dvl-tum/GHOST/blob/755a5dacfcf4dd122a4cac73061b24e9c84f3c19/config/config_tracker_dance.yaml)：`active/inactive appearance config`.
- [config/config_tracker_bdd.yaml](https://github.com/dvl-tum/GHOST/blob/755a5dacfcf4dd122a4cac73061b24e9c84f3c19/config/config_tracker_bdd.yaml)：`active/inactive appearance config`.
- [config/config_tracker_20.yaml](https://github.com/dvl-tum/GHOST/blob/755a5dacfcf4dd122a4cac73061b24e9c84f3c19/config/config_tracker_20.yaml)：`active/inactive appearance config`.
- [src/base_tracker.py](https://github.com/dvl-tum/GHOST/blob/755a5dacfcf4dd122a4cac73061b24e9c84f3c19/src/base_tracker.py)：`inactive threshold setup`.

**源码事实：** past_feats 可用 last/mean/median/moving average，each_sample 距离再聚合；active/inactive 可同轮或分轮指派，阈值与耐心分开。

**JEV 推论：** 先确定 anchor 表示与聚合，再比较 learnable retention；相机/年龄状态不能混入绝对 frame。

**不能照搬：** 源码同时保留诊断 gt_id 日志，这些字段不能进入我们的 runtime controller；各数据集配置阈值不可视为通用值。

## 17. dvl-tum/SUSHI

Commit：`ff1952b408835007f07d1fc78760872625fa6ae4`。

- [src/models/hiclnet.py](https://github.com/dvl-tum/SUSHI/blob/ff1952b408835007f07d1fc78760872625fa6ae4/src/models/hiclnet.py)：`HICLNet.__init__`, `HICLNet.forward`.
- [src/models/mpntrack.py](https://github.com/dvl-tum/SUSHI/blob/ff1952b408835007f07d1fc78760872625fa6ae4/src/models/mpntrack.py)：`MOTMPNet.forward`, `TimeAwareNodeModel.forward`, `EdgeModel.forward`.
- [src/tracker/hicl_tracker.py](https://github.com/dvl-tum/SUSHI/blob/ff1952b408835007f07d1fc78760872625fa6ae4/src/tracker/hicl_tracker.py)：`HICLTracker.hicl_forward`, `HICLTracker._project_graph`.
- [src/data/graph.py](https://github.com/dvl-tum/SUSHI/blob/ff1952b408835007f07d1fc78760872625fa6ae4/src/data/graph.py)：`HierarchicalGraph._get_curr_graph_specs`, `HierarchicalGraph.construct_curr_graph_nodes`, `HierarchicalGraph.add_edges_to_curr_graph`, `HierarchicalGraph.update_maps_and_depth`.

**源码事实：** 多层时间图由检测聚成 tracklet，逐层重建 edge/context；配置可共享全部、除首层或不共享权重，层级 embedding 与 solver projection 独立。

**JEV 推论：** 统一 reasoning 不等于统一 horizon；typed lifecycle 的共享 core 与问题 adapter 有代码层面的可行参照。

**不能照搬：** SUSHI 的图使用整段/子段过去和未来，是 offline tracker；不得直接把跨未来图特征用于在线 JEV 或把 binary edge classifier 称为无结构。

## 18. dvl-tum/mot_neural_solver

Commit：`4541eb605a922876c376ec95d4e55509175d145e`。

- [src/mot_neural_solver/models/mpn.py](https://github.com/dvl-tum/mot_neural_solver/blob/4541eb605a922876c376ec95d4e55509175d145e/src/mot_neural_solver/models/mpn.py)：`MOTMPNet.forward`, `MOTMPNet._build_core_MPNet`, `EdgeModel.forward`, `TimeAwareNodeModel.forward`.
- [src/mot_neural_solver/tracker/projectors.py](https://github.com/dvl-tum/mot_neural_solver/blob/4541eb605a922876c376ec95d4e55509175d145e/src/mot_neural_solver/tracker/projectors.py)：`GreedyProjector.project`, `ExactProjector.project`, `PuLPMinCostFlowSolver._add_constraints`, `PuLPMinCostFlowSolver.solve`.

**源码事实：** node/edge encoder 后多轮 message passing；binary edge logits 之后由 greedy/LP projection 限制每节点入/出flow<=1。

**JEV 推论：** 是否结构化要看 candidate interaction、state transitions 与约束的有效贡献，不能只数动作类别；G2/G3/G4 闭环消融为必要证据。

**不能照搬：** 该 solver 是跨时间 offline图，且仅 scalar edge classifier 的部分并不包含完整结构；不能移植未来节点或规模较大的图训练。

## 19. ShuCvlab/DiffMOT

Commit：`eada72e74e54c153b30674d277b97882edc568b8`。

- [tracker/DiffMOTtracker.py](https://github.com/ShuCvlab/DiffMOT/blob/eada72e74e54c153b30674d277b97882edc568b8/tracker/DiffMOTtracker.py)：`STrack.multi_predict_diff`, `STrack.re_activate`, `STrack.update`, `diffmottracker.update`, `STrack.__init__`.

**源码事实：** 运动条件维护观测/预测混合历史；diffusion 预测后恢复时用实际观测替换 condition；appearance 使用 confidence adaptive alpha，lost/new 生命周期仍单独管理。

**JEV 推论：** 不同 lifecycle state 需要不同相对历史；counterfactual branches 必须使用同一 RNG/未来 policy。

**不能照搬：** diffusion sampling 和运动网络的收益不能算 controller 收益；branch 分别抽样会破坏干预归因，不能照搬未冻结随机预测。

## 20. MCG-NJU/MOTIP

Commit：`ffc0e905ac196a603027eca8d18fb0dff48c8bcc`。

- [models/runtime_tracker.py](https://github.com/MCG-NJU/MOTIP/blob/ffc0e905ac196a603027eca8d18fb0dff48c8bcc/models/runtime_tracker.py)：`RuntimeTracker.update`, `RuntimeTracker._get_id_pred_labels`, `RuntimeTracker._assign_newborn_id_labels`, `RuntimeTracker._update_trajectory_infos`, `RuntimeTracker._filter_out_inactive_tracks`, `RuntimeTracker._hungarian_assignment`.
- [models/motip/trajectory_modeling.py](https://github.com/MCG-NJU/MOTIP/blob/ffc0e905ac196a603027eca8d18fb0dff48c8bcc/models/motip/trajectory_modeling.py)：`TrajectoryModeling.forward`.
- [models/motip/id_decoder.py](https://github.com/MCG-NJU/MOTIP/blob/ffc0e905ac196a603027eca8d18fb0dff48c8bcc/models/motip/id_decoder.py)：`IDDecoder.forward`, `IDDecoder._forward_a_layer`, `IDDecoder.generate_empty_id_embed`.
- [train.py](https://github.com/MCG-NJU/MOTIP/blob/ffc0e905ac196a603027eca8d18fb0dff48c8bcc/train.py)：`prepare_for_motip`, `train_one_epoch identity criterion path`.

**源码事实：** trajectory_features/boxes/IDs/times/masks 参与 ID decoder，causal attention 禁止当前及未来历史；newborn token 可重复为多个出生列，Hungarian 维护旧 ID 唯一指派；miss_tolerance 限制历史。

**JEV 推论：** old-vs-new prediction 必须放入 trajectory/candidate context；出生占位与禁止当前 edge 的 REASSOCIATE 都应检查约束语义。

**不能照搬：** 这是可训练的大 detection/attention 系统，不能在小 controller 中照搬；训练 annotation ID只用于 target，绝不能转成 runtime GT feature。

## 审计对 Phase VI 的直接约束

- 先执行 scalar/dynamic/binary 与 B2 去 REASSOCIATE 的闭环消融。三动作名称不能单独证明结构化；binary graph reasoning 也可能是结构化方法。
- MEMORY 的 intervention 必须跟踪其 anchor 直到 stale-bank 真正读取。未读取、被后续 write 覆盖或序列末尾截断分别记录；未读取样本权重0。
- Reactivation future rollout 使用 B2 MATCH + 即时 GMT MEMORY/REACT，禁止用固定 OFF action category 覆盖分支状态。
- fixed EMA alpha(old) 与 QD memo momentum(new) 分别实现，gallery 的 mean/min aggregation 必须显式声明。
- state corruption 修改真实过去状态并重新生成 supervised utility；不能只给 feature tensor 加噪音后保留失效旧 target。
- typed adapters 接共享 core，但 MATCH 保留 B2 相同64D输入；MEMORY/REACT采用相对年龄、候选与质量字段，GT/future仅用于离线 target。
- SUSHI、AFLink、Neural Solver 的 offline 未来图/轨迹后处理不得进入 online state。


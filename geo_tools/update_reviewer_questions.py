"""Refresh reviewer evidence from current artifacts; pending gates stay explicit."""
import csv
from common import *

def csvmap(path):
    if not path.exists():return {}
    return {r['setting']:r for r in csv.DictReader(path.open())}

def pct(row,key='recall1'):
    x=row.get(key);return 'missing' if not x else f'{float(x)*100:.2f}%'

def main():
    suite=json.loads((ROOT/'results/B0_strict_suite_v1_metrics.json').read_text());hard=csvmap(ROOT/'results/hard_feature_screening.csv');native=csvmap(ROOT/'results/hard_feature_native.csv')
    baseline=hard.get('gmt_reid_cosine',{});g17=hard.get('vggt/layer_17/roi_align',{});n17=native.get('vggt/layer_17/roi_align',{})
    retrieval=list((RUNTIME/'retrieval').glob('*/retrieval.metrics.json'))
    full_retrieval=[path for path in retrieval if json.loads((path.parent/'manifest.json').read_text()).get('full_video')]
    retrieval_evidence=f'实际在线图像读取、GMT目标查询与轨迹片段索引已实现，已完成{len(retrieval)}个检索评价，其中{len(full_retrieval)}个为完整视频。详见retrieval_results.csv；GT只进入事后评价，不进入查询选择/排序。'
    query_audit=ROOT/'results/retrieval_paired_query_outcomes_v1.json'
    if query_audit.exists():
        paired=json.loads(query_audit.read_text());candidate=next((row for row in paired['rows'] if row['method']=='VGGT2'),None)
        if candidate:
            retrieval_evidence+=f" 已完成denseB7跟踪接入原生排序的完整两路检索：相同{candidate['common_eligible']}个可检索查询中Recall1修正{candidate['common_eligible_rank1_fixes']}、破坏{candidate['common_eligible_rank1_harms']}，可检索集合变化{candidate['eligible_gained']}/{candidate['eligible_lost']}。这是小型主种子信号，匹配容量跟踪发布对照分别报告，已声明逐查询对照是否完成见审计状态，不能宣称VGGT独有或稳定收益。"
        for row in paired['rows']:
            if row['method'].startswith(('GMT-head','DINO-head')):
                retrieval_evidence+=f" 已完成{row['method']}的scene{row['query_scene']}：{row['common_eligible']}共同可检索/{row['requests']}请求，R1修正/破坏{row['common_eligible_rank1_fixes']}/{row['common_eligible_rank1_harms']}，MRR变化{row['common_eligible_mrr_delta']:+.6f}，首个正样本排名/可检索集合变化查询数{row['changed_rank_or_eligibility_queries']}；这不证明完整排名列表逐项相同。"
            if row['method']=='VGGT4':
                retrieval_evidence+=f" 四路VGGT完整回放scene{row['query_scene']}：{row['common_eligible']}个共同可检索查询，R1修正/破坏{row['common_eligible_rank1_fixes']}/{row['common_eligible_rank1_harms']}，MRR变化{row['common_eligible_mrr_delta']:+.6f}，可检索集合变化{row['eligible_gained']}/{row['eligible_lost']}。scene18修正的是两路实验中的同一查询，不是独立重复证据。"
        for row in paired.get('shared_control_corrections',[]):
            if row['vggt_rank1_fixes']:
                retrieval_evidence+=f" 同一有序查询核验：{row['vggt_method']}/{row['control_method']}在scene{row['query_scene']}分别修正{row['vggt_rank1_fixes']}/{row['control_rank1_fixes']}条，共同修正{row['shared_rank1_fixes']}条；VGGT有{row['vggt_fixes_not_reproduced_by_this_control']}条未被此对照复现。聚合提升相同不等于纠正同一查询，单次不同查询纠正也不证明稳定独有信息。"
        prior_queries=next((row for row in paired['rows'] if row['method']=='Prior4' and row['query_scene']==1),None)
        if prior_queries:
            retrieval_evidence+=f" 四路scene1固定先验的逐查询R1修正/破坏为{prior_queries['common_eligible_rank1_fixes']}/{prior_queries['common_eligible_rank1_harms']}；汇总相同不代表查询相同。"
    matched_gate=ROOT/'results/matched_head_tracking_retrieval_prefix_precheck_v2.json'
    if matched_gate.exists() and json.loads(matched_gate.read_text())['status']=='PASS':
        retrieval_evidence+=' GMT/DINO匹配跟踪头的真实40组交错前缀已在两个场景逐条复现独立参考。原核验器因两边权重均为0时未执行的几何方法字段差异而失败；只读v2显式记录该差异，仍核验全部生效配置、实际编码次数和精确输出。原失败/跳过记录保留，完整对照已固定恢复计划，仍未完成；短前缀不是检索收益证据。'
    signal_path=ROOT/'results/causal_B7_dense_positive_scene1_v1.json'
    signal=json.loads(signal_path.read_text()) if signal_path.exists() else None
    signal_text='尚无完整因果视频正向信号。'
    if signal:
        delta=signal['rows'][0]['metric_delta']
        signal_text=f"完整因果scene1每帧B7出现初步身份收益：CVIDF1 {delta['CVIDF1']:+.4f}个百分点、AssA {delta['AssA']:+.4f}、IDSW {delta['IDSW']:+d}；逐项检测完全一致。完整strict5及VGGT/DINO/GMT头和几何独立作用尚待确认，不能外推为稳定VGGT或三模块贡献。"
    dense_path=ROOT/'results/causal_B7_fixed_k1_strict_suite_v1_metrics.json'
    pair_path=ROOT/'results/causal_B7_dense_full_strict_identity_pair_audit_v1.json'
    if dense_path.exists() and pair_path.exists():
        dense=json.loads(dense_path.read_text());pairs=json.loads(pair_path.read_text())
        base=json.loads((ROOT/'results/causal_B0_strict_suite_v1_metrics.json').read_text())['metrics']
        if dense['status']=='PASS' and pairs['full_strict_suite'] and all(row['added_detections']==0 and row['removed_detections']==0 for row in pairs['rows']):
            signal_text=f"每帧B7完整strict5单种子CVIDF1={dense['metrics']['CVIDF1']:.6f}（相对因果B0 {dense['metrics']['CVIDF1']-base['CVIDF1']:+.6f}个百分点）、IDSW {base['IDSW']}→{dense['metrics']['IDSW']}，五场景检测key集合完全一致。身份关系审计同时记录修正和破坏；主种子完整DEV容量对照已完成，DINO数值上更高；重复密集稳定性未完成，不是稳定VGGT独有贡献。"
    camera_probe=ROOT/'results/inference_camera_encoding_source_probe_v1.json'
    cached_note=''
    constant_note=''
    constant_path=ROOT/'results/constant_native_head_scene1_accuracy_v1.json'
    if constant_path.exists():
        constant=json.loads(constant_path.read_text())['rows'][0]
        constant_note=f" 原生GMT-head固定+1配对分数完整scene1对照CVIDF1={constant['metrics']['CVIDF1']:.6f}、IDSW={constant['metrics']['IDSW']}，强于当前VGGT/DINO学习头scene1结果；该对照保留head编码用于有效性、丢弃学习配对分数，且traced速度无效。无额外head的等价实现已通过真实短前缀/零系数检查并完整复现scene1预测SHA；完整strict5与配对推理种子状态见下文；scene1不能单独支持全开发结论。"
    native_full_path=ROOT/'results/native_prior_seed20261014_strict_suite_v1_metrics.json'
    if native_full_path.exists():
        native_full=json.loads(native_full_path.read_text())
        if native_full['status']=='PASS':
            constant_note+=f" 无额外head固定先验完整strict5已完成：CVIDF1={native_full['metrics']['CVIDF1']:.6f}、IDSW={native_full['metrics']['IDSW']}，优于当前主种子denseVGGT完整DEV结果；全五场景检测key集合一致。配对推理种子未完成，不能归因于VGGT或声称稳定速度优势。"
    seed_path=ROOT/'results/native_prior_inference_seed_stability_v1.json'
    if seed_path.exists():
        seeds=json.loads(seed_path.read_text());number=len(seeds['completed_inference_seeds']);limits=seeds['paired_cvidf1_delta_range']
        constant_note=constant_note.replace('配对推理种子未完成',f'配对推理种子已核验{number}/3组')
        constant_note+=f" 实际推理RNG种子配对的CVIDF1增量范围{limits[0]:+.6f}至{limits[1]:+.6f}个百分点；这是冻结同一模型的推理随机性，与三个独立训练head种子不同。未知/破坏关系保留，不能用净增益替代完整纠错与损害统计。"
    dense_seed_path=ROOT/'results/causal_dense_seed_stability_v1.json'
    if dense_seed_path.exists():
        factors=json.loads(dense_seed_path.read_text())
        signal_text+=' 密集因果种子复核分开记录：'+'；'.join(f"{group['method']}/{group['factor']}完成{group['completed_trials']}/3" for group in factors['groups'])+'。B7主种子在两类因素间共享，不是独立重复；完整原生strict5、实际推理种子及head哈希逐项核验，见GEO_STREAM_DENSE_SEED_STABILITY.md。'
        feature=next((group for group in factors['groups'] if group['method']=='B5' and group['factor']=='training_head_seed'),None)
        if feature and feature['complete']:
            signal_text+=' 特征单独启用B5的完整strict5三个TRAIN头CVIDF1增量为'+','.join(f"{row['paired_metric_delta']['CVIDF1']:+.6f}" for row in feature['rows'])+'个百分点，第三个为负；固定实际推理种子与检测集合，不能宣称稳定特征改进。B7训练头及固定主头推理RNG的完成状态分别见种子审计。'
        joint=next((group for group in factors['groups'] if group['method']=='B7' and group['factor']=='training_head_seed'),None)
        if joint and joint['complete']:
            signal_text+=' 联合B7完整strict5三个TRAIN头也全部完成，CVIDF1增量为'+','.join(f"{row['paired_metric_delta']['CVIDF1']:+.6f}" for row in joint['rows'])+'个百分点；第三头CVIDF1下降但IDSW减少，不能用IDSW单项改善替代身份F1或宣称均匀正向。未重新选择head、系数或评测划分。固定主头推理RNG复核仍单列。'
        observed=[row for row in factors.get('scene1_training_head_diagnostic',{}).get('rows',[]) if row['method']=='B7']
        if len(observed)==3:
            signal_text+=' 完整scene1的三个TRAIN头在固定推理种子20261014下CVIDF1分别为'+','.join(f"{row['metrics']['CVIDF1']:.6f}" for row in observed)+'，第三头低于同场景基线；检测集合和其他融合设置相同。这是单场景训练头观察，完整DEV结果另列，不能据此推断固定主头的推理RNG效应。'
    geometry_note='dense场景1特征单独启用与联合预测完全相同，三个几何通过门槛事件并无独立身份贡献。'
    geometry_path=ROOT/'results/causal_dense_geometry_full_strict_attribution_v1.json'
    if geometry_path.exists() and json.loads(geometry_path.read_text())['status']=='EXACT_FEATURE_JOINT_FULL_STRICT_OUTPUTS':
        geometry_note='主种子dense特征单独启用B5与联合B7在完整strict5的每条预测均完全一致；所有共同有效配置、head和编码调用次数已核验。联合几何共3次通过数值/背景门槛，均在scene1，但没有独立输出贡献。其他head的完整证据分别报告；该结果不证明所有几何无用或GT位姿准确。'
    for seed in (20261015,20261016):
        path=ROOT/f'results/causal_dense_geometry_seed{seed}_full_strict_attribution_v1.json'
        if path.exists():
            value=json.loads(path.read_text())
            if value['full_strict_suite'] and value['status']=='EXACT_FEATURE_JOINT_FULL_STRICT_OUTPUTS':
                geometry_note+=f" TRAIN头{seed}在固定推理RNG{value['inference_seed']}下也通过完整strict5的B5/B7精确输出一致性；本次运行有{sum(row['joint_valid_geometry_calls'] for row in value['rows'])}次几何数值/背景门槛通过，未改变输出。不同head重放相同场景不是独立物理几何样本。"
    failure_path=ROOT/'results/geometry_failure_taxonomy_v1.json'
    if failure_path.exists():
        failure=json.loads(failure_path.read_text());count=sum(row['diagnostic_rows'] for row in failure['summaries'])
        geometry_note+=f' 已整理{count}条固定几何筛选记录的实际门槛失败与未记录字段；重复遮罩/特征分支不是独立样本。内点不足或残差大不证明低重叠、低纹理或快速旋转；原始dirty源码与可恢复文件SHA分别保留，见GEO_STREAM_GEOMETRY_FAILURES.md。'
    cached_path=ROOT/'results/target_cache_three_backbone_scene1_controls_v1.json'
    if cached_path.exists() and json.loads(cached_path.read_text())['all_exact_baseline_predictions']:
        cached_note=' 新增有界目标特征复用的VGGT/DINO/GMT完整scene1对照均与baseline逐条一致，未保留dense收益；三个骨干相同40个观测更新时间，但实际VGGT80/DINO40/GMT0次编码，不能称同等计算。'
    camera_note=''
    if camera_probe.exists():
        probe=json.loads(camera_probe.read_text())
        if probe['status']=='INFERENCE_CAMERA_COLLAPSE_OBSERVED':
            camera_note=' 另发现官方单图推理空间/相机MLP输入的相机标量两路都为0（真实ROI方法语义探针，感知/MLP捕获为stub）；显式正确相机索引的实际baseline诊断已排队，锁定生产基线未变。不能在排除此来源前将增益归因于VGGT独有信息。'
            accuracy_path=ROOT/'results/camera_index_corrected_scene1_accuracy_audit_v1.json'
            if accuracy_path.exists():
                accuracy=json.loads(accuracy_path.read_text())
                if accuracy['exact_serialized_predictions']:
                    camera_note=' 官方单图推理相机MLP输入两路均为0的源码问题已确认；实际scene1正确相机索引对照与原baseline逐条预测完全一致、所有指标差值为0。因此该修正未解释scene1收益，其他完整strict场景仍须验证。锁定生产基线保留，源码问题与准确率影响分别报告。'
    full_camera_path=ROOT/'results/camera_index_corrected_full_strict_audit_v2.json'
    if full_camera_path.exists():
        full_camera=json.loads(full_camera_path.read_text())
        changed=sum(row['changed_track_id_rows'] or 0 for row in full_camera['rows'])
        camera_note=f' 正确相机索引对照已完成strict5：四场景逐条预测完全一致，scene18有{changed}条track ID变化，检测行不变，全部汇总指标不变。不能称完整输出等价。完整GT匹配身份关系审计修正/破坏均为0，但该条件结果不抹去原始ID变化；锁定生产基线保留。实际MLP输入与整个GMT相机换序的短序列探针独立排队。'
    hook_path=ROOT/'results/live_camera_index_hook_probe_v1.json'
    if hook_path.exists() and json.loads(hook_path.read_text())['status']=='PASS':
        camera_note=camera_note.replace('实际MLP输入与整个GMT相机换序的短序列探针独立排队。','真实checkpoint/图像前四组的8次MLP输入探针已通过：相机列0/0变为0/.5，其他十列逐项不变，第二路MLP输出确实改变。整个GMT换序另行诊断，不能用编码器换序代替。')
    whole_order_path=ROOT/'results/live_whole_gmt_camera_order_v1.json'
    if whole_order_path.exists():
        camera_note+=' 整个GMT实际scene1前40组换序检查发现OFF与B7均有身份划分变化，恢复物理ID并允许全局重编号后仍不同；不能宣称内部处理顺序无关。规范顺序逐条复现独立参考，检测集合相同，B7每版本80次VGGT调用、几何通过0。既有负例保留；物理相机输入适配器的独立验证见下文。'
    physical_path=ROOT/'results/physical_camera_input_contract_v1.json'
    if physical_path.exists():
        physical=json.loads(physical_path.read_text())
        if physical['status']=='PASS_EXPLICIT_CAMERA_INPUT_CONTRACT':
            camera_note+=f" 物理相机ID输入适配器的{len(physical['rows'])}项真实OFF/B7反序/交替顺序检查均通过，前{physical['max_frames']}组恢复后输出逐条等于规范参考，失败0。B7每版本80次VGGT调用、几何通过0；不证明内部顺序、完整视频或有效位姿不变性。"
    later_regression='缓存/诊断后的源码完整OFF/零系数回归已排队，按实际版本分别报告。'
    off_path=ROOT/'results/OFF_post_cache_trace_strict_v1_parity.json'
    zero_path=ROOT/'results/ZERO_post_cache_trace_strict_v1_parity.json'
    if off_path.exists() and json.loads(off_path.read_text())['status']=='PASS':
        later_regression='后续a9f7d1ec源码完整strict5 OFF逐条等价也已通过。'
        later_regression+=('同版本零系数完整回归也通过。' if zero_path.exists() and json.loads(zero_path.read_text())['status']=='PASS' else '同版本零系数完整回归仍在运行，尚未通过；不将旧版证据外推为当前全部源码验证。')
    source_check=ROOT/'results/current_default_source_consistency_v1.json'
    if source_check.exists():
        checked=json.loads(source_check.read_text())
        if checked['status']=='RECORDED_DEFAULT_SOURCE_MATCHES':
            later_regression+=f" 当前源码一致性核验保留{checked['controls_verified']}个完整OFF/零系数场景的原始预测和清单，每场景记录的{','.join(map(str,checked['matching_source_file_counts']))}个GTR/默认驱动文件与当前一致，checkpoint/YAML/TRAIN注释哈希相同。唯一改变的isolated_stream为检索专用；这不是新增推理，原始离线协议和未验证的可选检索范围保持明确，见current_default_source_consistency_v1.json。"
    questions=[
    ('为什么 RPCE 之外需要动态几何？','尚未证明需要。冻结相机几何在最小开发样本上未通过背景有效性门槛；传统几何对照和软平面分支已运行，不能据此声称新增动态几何有效。'+geometry_note+camera_note),
    ('为什么不直接使用 DINOv2/ReID？',f'必须保留强对照。12 个离线 GT 匹配困难查询上，GMT ReID Recall@1={pct(baseline)}，高于当前 VGGT 中间层。独立DINO普通融合的离线稀疏/逐帧完整严格开发对照和匹配容量学习头离线k10对照已完成；主种子因果dense匹配容量GMT/DINO完整strict5均已完成；DINO数值上高于当前VGGT联合版本，见对应完整suite指标。scene1的DINO匹配VGGT CVIDF1且少1次IDSW。证据：B1_dinov2_dense_strict_suite_v1_metrics.json、C_lowrank_gmt_seed14_k10_strict_v1_suite_metrics.json、C_lowrank_dinov2_seed14_k10_strict_v1_suite_metrics.json。'+constant_note),
    ('VGGT 多层是否提供独立身份信息？',f'未证明稳定或VGGT独有增益。第17层 ROI 的256维与原始维度 Recall@1 分别为 {pct(g17)}、{pct(n17)}。另在45个固定TRAIN查询中，原维度VGGT修正1个GMT ReID/提案错误，但破坏20个原正确ReID查询；同一修正也可由独立DINO完成。有限互补允许小头原型验证，不能宣称几何身份信息独占。'),
    ('为什么不简单拼接所有输出？','简单四层平均相似度等价于等权拼接归一化目标向量；B3完整严格开发输出与B0完全一致。B4标量weight2完整对照略低于B0。原始离线稀疏k10主种子GCIR未建立收益；因果dense有初步收益但固定先验更强。小头保留32维/两顺序64维目标表示，减少新增向量体积，但原GMT记忆和完整系统成本须分别测量。'),
    ('与 VGGT-Det 的差别？','当前实现做目标身份候选残差，而非3D检测；未复制其MMDetection3D/PointNet检测头。冻结多层聚合思想已有先例，任务不同本身不足以证明方法创新。见 GEOMETRY_OPEN_SOURCE_AUDIT.md。'),
    ('与已有 VGGT+ReID+3D tracking 的差别？','待证明的区别是动态几何有效性、身份表示互补和因果计算分配。已有校准自由室内方法使用VGGT/OsNet/姿态提升和聚类，简单几何+ReID已是先例。三个新增假设尚未取得完整证据。'),
    ('运动无人机如何保证几何一致性？','不比较不同调用的3D坐标。当前用单次联合调用相对相机关系，独立背景检查，失败即中性。合成相机投影和传播方向测试通过；此前稀疏完整对照均未通过门槛，每帧scene1有3次通过独立背景门槛，但没有GT位姿准确性或独立跟踪收益证明。'),
    ('少重叠视野是否有效？','没有此类有效性声明。无纹理、无足够可靠匹配、低覆盖、非有限或过期状态均为中性。ORB/SIFT和平面分支给出了实际低有效率，不强行拒绝身份候选。'),
    ('身份触发优于固定周期吗？','未证明。完整strict5固定/联合各400次真实VGGT调用：固定仅+.001627 CVIDF1个百分点、IDSW不变，联合与baseline一致；未建立稳定方法收益或联合优越性。固定k20无总量上限调用720次，指标与400次上限相同，没有额外身份收益。运动单一触发完整strict5已完成：既定阈值下0次选择、0次VGGT调用、预测指标同B0，实际预算不匹配400次固定对照，不能声称调度优势或不存在运动。身份单触发完整strict5也已完成，CVIDF1仅+.006508个百分点、IDSW−8，实际218次调用；不与400次固定/联合构成相同实际预算比较。其他dense间隔和重复矩阵仍在完成。证据：causal_B8_B9_complete_budget_runtime_audit_v1.json、causal_k20_uncapped_vs_cap80_budget_runtime_v1.json。'+cached_note),
    ('相较 StreamVGGT/FastVGGT 的价值？','不能直接移用长序列加速倍率或流式权重。源码接口和许可证已审计；真实三组TRAIN双视角微基准v2已执行：original36调用通过，Fast在V100的BF16输出进入相机head LayerNorm时失败；NO_VALID_SPEED_COMPARISON，不能把进程退出0当成方法成功。相机输入FP32的v3通过但Fast保留BF16、仍较慢；另在独立源码中仅替换10个dtype常量的完全FP16路径v4也通过各36调用，三组original/Fast比为.465760/.469999/.471315，当前短双视角V100实现仍慢约2.12–2.15倍，且特征改变，因此不接入当前跟踪。未改合并比例/算法或训练权重，不外推长序列/A100/完整系统。'),
    ('完整 FPS/延迟/显存/参数量？','因果B0 scene18完整实测6.64图像/s；优化异步相机路径5.40图像/s且预测完全相同，0有效几何。实际硬件为V100，不能称A100结果。静态GMT模型初始化排除，新增模型冷启动及worker启动/关闭算入总成本。实际CPU结构/411键严格加载核对GMT注册参数53,515,309，三个匹配头309,319/309,849/310,402。resource_ledger_v1分别记录CUDA allocator、主存RSS、最终原GMT历史和缓存/索引载荷；未测字段为null，不相加非同时峰值。原GMT历史仍不受索引容量上限限制。不以一次计时宣称稳定加速。'),
    ('提升是否来自不同检测/配置/评测 bug？',signal_text+f' 固定checkpoint/config与映射修复。完整离线五段严格基线HOTA={suite["metrics"]["HOTA"]:.4f}、CVIDF1={suite["metrics"]["CVIDF1"]:.4f}，不能拿离线数值替代因果对照。原矩阵c214源码完整五场景OFF与零系数逐条预测等价均通过；'+later_regression+camera_note),
    ('三个模块能否分别证明有效？','尚未证明。'+signal_text+constant_note+geometry_note+' 当前稳定科学成功/ENGINEERING_COMPLETE/PAPER_READY均未达成，完整消融、因果速度和重复复核仍须完成。'),
    ('为什么符合 WWW？','仍有实质适配风险。官方CFP要求明确Web科学问题，仅用数据集或API不充分。'+retrieval_evidence+' 本地数据集回放不等于互联网来源、真实用户查询或Web部署；见WWW_SCOPE_NOTE.md。'),
    ('在线视频关联检索的真实贡献？',retrieval_evidence+' 条件Recall@K、完整MRR、检测覆盖、可检索比例、全部请求成功率和本地查询延迟分别报告。完整读帧/检测/GTA/新增特征/索引/查询成本另计；流数量与关键帧预算对照及实质Web系统贡献仍需完成。')]
    stress_path=ROOT/'results/constructed_geometry_degeneracy_v1.json'
    if stress_path.exists():
        stress=json.loads(stress_path.read_text())
        if stress['status']=='PASS_CONSTRUCTED_CPU_GEOMETRY_STRESS':
            note=' 构造CPU退化检查另行通过：空白/独立生成背景的ORB/SIFT均拒绝几何并原样保留分数；60°平面旋转使光流失效、旧几何禁用，但运动单触发未请求更新。因此motion=0可表示估计失败，不代表无运动。这些是合成检测/过去F条件下的代码检查，不是真实UAV视野/旋转或VGGT有效性证据；见constructed_geometry_degeneracy_v1.json。'
            selected={'运动无人机如何保证几何一致性？','少重叠视野是否有效？','身份触发优于固定周期吗？'}
            questions=[(q,a+note if q in selected else a) for q,a in questions]
    process_path=ROOT/'results/process_wall_runtime_v1.json'
    if process_path.exists():
        process=json.loads(process_path.read_text())
        note=f" 另有{len(process['rows'])}个完整进程、{len(process['suites'])}套完整strict5的外层启动到退出观察耗时；{len(process['excluded_held_stages'])}个与调度器暂停窗口关联的条目被排除。进程口径包含Python导入、静态模型准备及导出/溯源，单独GT评价和队列等待不计入；区间差额不是纯模型初始化。缺失不补0，详见GEO_STREAM_PROCESS_WALL_RUNTIME.md。"
        questions=[(q,a+note if q=='完整 FPS/延迟/显存/参数量？' else a) for q,a in questions]
    lines=['# Reviewer questions — current evidence','', 'This file is generated from current artifacts; pending items are not claims of completion.','']
    for i,(q,a) in enumerate(questions,1):lines += [f'## Q{i}: {q}','',a,'']
    (ROOT/'docs/REVIEWER_QUESTIONS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()

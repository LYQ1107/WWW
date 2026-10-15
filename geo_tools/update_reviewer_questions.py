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
        camera_note+=' 整个GMT实际scene1前40组换序检查发现OFF与B7均有身份划分变化，恢复物理ID并允许全局重编号后仍不同；不能宣称内部处理顺序无关。规范顺序逐条复现独立参考，检测集合相同，B7每版本80次VGGT调用、几何通过0。物理相机ID输入适配器会固定内部顺序并恢复调用方输出位置，CPU测试通过、实际反序/交替顺序验证待完成；既有负例保留。'
    later_regression='缓存/诊断后的源码完整OFF/零系数回归已排队，按实际版本分别报告。'
    off_path=ROOT/'results/OFF_post_cache_trace_strict_v1_parity.json'
    zero_path=ROOT/'results/ZERO_post_cache_trace_strict_v1_parity.json'
    if off_path.exists() and json.loads(off_path.read_text())['status']=='PASS':
        later_regression='后续a9f7d1ec源码完整strict5 OFF逐条等价也已通过。'
        later_regression+=('同版本零系数完整回归也通过。' if zero_path.exists() and json.loads(zero_path.read_text())['status']=='PASS' else '同版本零系数完整回归仍在运行，尚未通过；不将旧版证据外推为当前全部源码验证。')
    questions=[
    ('为什么 RPCE 之外需要动态几何？','尚未证明需要。冻结相机几何在最小开发样本上未通过背景有效性门槛；传统几何对照和软平面分支已运行，不能据此声称新增动态几何有效。dense场景1特征单独启用与联合预测完全相同，三个几何通过门槛事件并无独立身份贡献。'+camera_note),
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
    ('三个模块能否分别证明有效？','尚未证明。'+signal_text+constant_note+' 当前稳定科学成功/ENGINEERING_COMPLETE/PAPER_READY均未达成，完整消融、因果速度和重复复核仍须完成。'),
    ('为什么符合 WWW？','仍有实质适配风险。官方CFP要求明确Web科学问题，仅用数据集或API不充分。'+retrieval_evidence+' 本地数据集回放不等于互联网来源、真实用户查询或Web部署；见WWW_SCOPE_NOTE.md。'),
    ('在线视频关联检索的真实贡献？',retrieval_evidence+' 条件Recall@K、完整MRR、检测覆盖、可检索比例、全部请求成功率和本地查询延迟分别报告。完整读帧/检测/GTA/新增特征/索引/查询成本另计；流数量与关键帧预算对照及实质Web系统贡献仍需完成。')]
    lines=['# Reviewer questions — current evidence','', 'This file is generated from current artifacts; pending items are not claims of completion.','']
    for i,(q,a) in enumerate(questions,1):lines += [f'## Q{i}: {q}','',a,'']
    (ROOT/'docs/REVIEWER_QUESTIONS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()

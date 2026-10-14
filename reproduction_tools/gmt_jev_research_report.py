"""Build current evidence reports without choosing runs or declaring goal completion."""
from gmt_jev_common import *
from gmt_jev_job import spent

KINDS=['original','threshold','rule','confidence','mlp_zero','attention_zero','cic_zero']

def main():
 online=read(REPORTS/'ONLINE_METRICS.json');completed={};pending=[]
 kinds=KINDS+(['rule_cached'] if (REPORTS/'SPEED_rule_cached.json').exists() else [])
 for kind in kinds:
  p=REPORTS/f'SPEED_{kind}.json'
  if not p.exists():pending.append(kind);continue
  r=read(p);assert r['status']=='COMPLETE' and len(r['trials'])==3
  assert r['physical_GPU']==7 and r['no_GTA_or_RPCE_pruning'] and r['no_perception_cache']
  completed[kind]=dict(report=ref(p),GPU_UUID=r['GPU_UUID'],mean_scene_FPS=r['mean_full_scene_FPS'],
   mean_image_FPS=2*r['mean_full_scene_FPS'],extra_parameters=r['extra_trainable_parameters'],
   full_system_parameters=r['full_system_parameters'],trials=r['trials'],untrained_prototype=r['untrained_prototype'],stock_prediction_SHA=r['stock_prediction_SHA'])
 assert len({r['GPU_UUID'] for r in completed.values()})<=1
 if 'original' in completed:
  base=completed['original']['mean_scene_FPS']
  for kind,r in completed.items():
   r['FPS_ratio_to_GMT']=r['mean_scene_FPS']/base;r['efficiency_gate_pass']=r['FPS_ratio_to_GMT']>=.95
   assert r['full_system_parameters']==completed['original']['full_system_parameters']+r['extra_parameters']
   if r['untrained_prototype']:
    r['full_video_stock_prediction_parity_with_original']=r['stock_prediction_SHA']==completed['original']['stock_prediction_SHA']
    assert r['full_video_stock_prediction_parity_with_original'],kind+' zero residual changes stock output'
 if 'rule_cached' in completed:
  proof=read(REPORTS/'FEATURE_CACHE_PARITY.json')
  assert proof['status']=='PASS_EXACT_FULL_NATIVE_RULE' and proof['frames']==1034 and proof['score_feature_pairs']==2067
  cached=completed['rule_cached'];uncached=completed['rule']
  cached['full_video_stock_prediction_parity_with_uncached_rule']=cached['stock_prediction_SHA']==uncached['stock_prediction_SHA']
  assert cached['full_video_stock_prediction_parity_with_uncached_rule'],'Feature cache changes native stock predictions'
  cached['feature_cache_parity_report']=ref(REPORTS/'FEATURE_CACHE_PARITY.json')
  cached['FPS_ratio_to_uncached_rule']=cached['mean_scene_FPS']/uncached['mean_scene_FPS']
 speed=dict(status='COMPLETE' if not pending else 'PENDING_REAL_MEASUREMENTS',methods=completed,pending=pending,
  conventions=dict(scene='One synchronized two-camera time step',image='Two image inputs per scene; image FPS = 2 * scene FPS',
   stage2='Actual GTA through native commit, with shared CUDA synchronization instrumentation. Per-trial p50/p95, excluding first16 events.',
   full='Real image IO/resize, detector/VFCE/RPCE/GTA, native association/history, original stock filtering and postprocess; model load/warmup excluded.',
   scope='Same GPU7, three reset full-video repeats per method, no GT observer. Serial method blocks on a shared host; timing variability remains a limitation.'),
  trained_neural_model_quality='NOT_ESTABLISHED_NO_NEW_TRAINING',source=binding())
 cost=REPORTS/'HEAD_COST_PREFIX.json'
 if cost.exists():speed['dense_operation_audit']=ref(cost)
 save(REPORTS/'SPEED_METRICS.json',speed)
 stock_path=REPORTS/'ORIGINAL_GMT_STOCK_METRICS.json'
 stock=None
 stock_temporal=REPORTS/'STOCK_IDENTITY_TEMPORAL_AUDIT.json'
 if stock_path.exists():
  stock=read(stock_path);online['matched_original_stock_baseline']=dict(report=ref(stock_path),metrics=stock['metrics'],CVIDF1=stock['CVIDF1'])
  for kind,r in online['methods'].items():
   s=r['secondary_stock'];r['secondary_stock_delta']={k:s['metrics'][k]-stock['metrics'][k] for k in s['metrics']}
   r['secondary_stock_delta']['CVIDF1']=s['CVIDF1']-stock['CVIDF1']
   r['secondary_stock_metric_gates']=dict(HOTA=s['metrics']['HOTA']>=stock['metrics']['HOTA']+.5,
    AssA=s['metrics']['AssA']>=stock['metrics']['AssA'],CVIDF1=s['CVIDF1']>=stock['CVIDF1'],
    IDSW=s['metrics']['IDSW']<=stock['metrics']['IDSW']*1.05,Frag=s['metrics']['Frag']<=stock['metrics']['Frag']*1.05,
    long_identity_duration='PENDING_MATCHED_STOCK_DIAGNOSTIC')
   if stock_temporal.exists():
    r['secondary_stock_identity']=read(stock_temporal)['methods'][kind]
    r['secondary_stock_metric_gates']['long_identity_duration']=r['secondary_stock_identity']['retrospective_duration_ratio_to_original']<=1.05
    r['secondary_stock_post_hoc_past_anchor_duration_ratio']=r['secondary_stock_identity']['past_anchor_duration_ratio_to_original']
  text=(DOCS/'ONLINE_RESULTS.md').read_text().split('\n## 同约定 stock 对照')[0]
  text=text.replace('原生 min_track_len=50 与 stock 后处理的结果保存在每条 ONLINE_*.json 的 secondary_stock；原始 GMT 的同约定基线尚待计时任务保留的输出评测，当前不计算 stock 差值、不混用 raw 基线。','原生 min_track_len=50 与 stock 后处理的匹配基线和对照另列于下表；不与 raw 混用，预注册主要协议仍为 raw。')
  table=['','## 同约定 stock 对照','','| 方法 | HOTA | ΔHOTA | AssA | CVIDF1 | IDSW | Frag |','|---|---:|---:|---:|---:|---:|---:|']
  cases={'original':stock,**{k:r['secondary_stock'] for k,r in online['methods'].items()}}
  for kind,r in cases.items():
   m=r['metrics'];table.append(f"| {kind} | {m['HOTA']:.3f} | {m['HOTA']-stock['metrics']['HOTA']:+.3f} | {m['AssA']:.3f} | {r['CVIDF1']:.3f} | {m['IDSW']:.0f} | {m['Frag']:.0f} |")
  (DOCS/'ONLINE_RESULTS.md').write_text(text+'\n'.join(table)+'\n')
 for kind,r in online['methods'].items():
  if kind in completed:r['FPS']=completed[kind]
 temporal=REPORTS/'PAST_ANCHOR_TEMPORAL_AUDIT.json'
 if temporal.exists():online['post_hoc_past_anchor_temporal_audit']=ref(temporal)
 save(REPORTS/'ONLINE_METRICS.json',online)
 decision=dict(status='NOT_ESTABLISHED',scientific_success=False,bounded_tracking_experiments_complete=True,
  efficiency_status=speed['status'],master_goal_status='ACTIVE',
  development_primary_conclusions={k:dict(delta_HOTA=r['delta']['HOTA'],quality_gates=r['development_quality_gates'],joint_quality_pass=r['all_quality_gates_pass']) for k,r in online['methods'].items()},
  learned_training='NOT_RUN_FROZEN_TRAIN_SIGNAL_GATE_FAILED',architecture_ablation='NOT_RUN_NO_JOINT_POSITIVE_MAIN_METHOD',
  current_best_conservative_candidate='confidence: modest ordinary calibration development signal, not +0.5 or JEV-specific superiority',
  rule_go_no_go='PRIMARY_RAW_NO_GO: HOTA +0.502582 but IDSW +15.44%; native stock assessed separately',
  unmet=['HOTA gain and identity stability jointly satisfied','Qualified learned MLP/Attention/CIC comparison','Independent full-video validation','Necessary seed reproduction'],
  prohibited_next_steps=['Tune on TRAIN16 signs','Read new DEV/sealed20-22/TEST/Full24','Replace original GTA','Blindly train larger JEV'],source=binding())
 if pending:decision['unmet'].append('Completed matched full-image efficiency measurements')
 if stock is not None:
  decision['secondary_stock_development_metric_gates']={k:r['secondary_stock_metric_gates'] for k,r in online['methods'].items()}
  decision['secondary_stock_interpretation']='Rule passes HOTA/AssA/CVIDF1/IDSW/Frag secondary metric gates; stock temporal diagnostics reported separately and speed remains method-specific. Threshold/confidence fail secondary IDSW. This does not override the preregistered primary raw outcome or establish scientific success.'
 if temporal.exists():
  d=read(temporal);decision['post_hoc_temporal_diagnostics']=dict(report=ref(temporal),
   ratios={k:r['duration_ratio_to_original'] for k,r in d['methods'].items()},
   does_not_replace_preregistered_gates=True,not_causal_propagation=True)
 save(REPORTS/'GO_NO_GO.json',decision)
 goal=read(REPORTS/'MASTER_GOAL.json');goal['status']='ACTIVE';goal['scientific_success']=False
 goal['gates']['SPEED']='PENDING_REAL_FULL_IMAGE_TRIALS' if pending else 'COMPLETE_MEASUREMENTS_SEE_PER_METHOD_GATES'
 goal['gates']['INDEPENDENT_VALIDATION']='PENDING_AUTHORIZED_UNSEEN_SOURCE';save(REPORTS/'MASTER_GOAL.json',goal)
 efficiency=['# 实测效率与参数','', '以下只使用完成的三次完整视频结果；缺测项仍为 PENDING。', '',
  '| 方法 | 场景 FPS | 图像 FPS | 相对 GMT | 新增参数 |', '|---|---:|---:|---:|---:|']
 for kind in kinds:
  if kind not in completed:efficiency.append(f'| {kind} | PENDING | PENDING | PENDING | PENDING |');continue
  r=completed[kind];ratio=r.get('FPS_ratio_to_GMT');ratio_text=f'{ratio:.4f}' if ratio is not None else 'PENDING'
  efficiency.append(f"| {kind} | {r['mean_scene_FPS']:.4f} | {r['mean_image_FPS']:.4f} | {ratio_text} | {r['extra_parameters']} |")
 efficiency+=['','每个场景时刻读取两张图像，图像 FPS 为场景 FPS 的两倍，相对效率比值一致。完整计时包含原生 min_track_len=50 和 stock 后处理；不包含模型加载与 8 帧 warmup。每次完整运行重置原生状态、随机状态和额外摘要，不复用 perception cache。GPU 型号为 V100-DGXS-32GB，实际 UUID 及每次 VRAM、Stage2 p50/p95 见 JSON。',
  '', '原型 MLP/Attention/CIC 均未训练，残差输出为零。这些数据只能说明集成与开销，不能说明训练后的质量、优势或安全性。Rule 没有可学习参数，但仍有跨相机特征和历史维护计算；extra_neural_FLOPs=0 不代表额外模块成本为零。', '',
  'HEAD_COST_PREFIX.json 若已完成，则使用真实 TRAIN16 前64场景帧的原生候选尺寸，记录 Linear 与 Attention 矩阵乘法，并单列特征余弦矩阵乘法。每个 MAC 算两 FLOPs；排除偏置、归一化、GELU、softmax、reduction、逐元素运算、Python 与数据移动，不能称作总 FLOPs 或完整视频总开销。', '',
  '计时使用同一 GPU 的串行方法块，宿主机仍有其他卡任务；CUDA 同步剖析开销各方法一致计入。三次原生输出必须逐次严格相同；95% 门槛是预注册开发指标，不等于独立种子验证。']
 (DOCS/'EFFICIENCY.md').write_text('\n'.join(efficiency)+'\n')
 if 'rule_cached' in completed:
  r=completed['rule_cached']
  with (DOCS/'EFFICIENCY.md').open('a') as f:
   f.write(f"\n可选历史摘要缓存已通过完整1034帧、2067次原生调用的特征/门控/残差/分数比较；三轮计时的最终 stock 预测指纹也与未缓存 Rule 严格一致。缓存版图像 FPS 为 {r['mean_image_FPS']:.4f}，相对 GMT 为 {r['FPS_ratio_to_GMT']:.4f}，相对未缓存 Rule 为 {r['FPS_ratio_to_uncached_rule']:.4f}。本次计时未观察到加速，未通过95%效率门槛；共享宿主机的串行计时不能单独分离归一化计算与系统负载的因果贡献。缓存等价不会改变既有 Rule 的主要协议身份稳定性失败结论。\n")
 if cost.exists():
  d=read(cost);extra=['','实际64场景帧、127次原生调用的 dense matmul 审计：','','| 原型 | head GFLOPs | 特征余弦 GFLOPs |','|---|---:|---:|']
  for kind,r in d['methods'].items():extra.append(f"| {kind} | {r['head_dense_matmul_FLOPs_total']/1e9:.6f} | {r['feature_cosine_matmul_FLOPs_total']/1e9:.6f} |")
  with (DOCS/'EFFICIENCY.md').open('a') as f:f.write('\n'.join(extra)+'\n')
 (DOCS/'REVIEWER_ANALYSIS.md').write_text('''# 科研证据与审稿问题

当前证据证明了完整 GMT 恢复和原生集成等价，未证明新架构的独立优势。原始 GTA 实际参与推理，VFCE+RPCE=1152 维；新模型只在 native summed candidate scores 之后添加有界残差，没有替代前端、Gallery、Bank 或求解器。

MCMT 的潜在研究问题是跨相机冲突、合法候选竞争以及身份状态的长期传播。已有冲突规则的 HOTA 正向并不足以证明这些信息被正确使用：完整视频 IDSW 明显退化，严格认证的局部 N01 没有正向。需要新的合格 TRAIN 信号和可复现闭环优势，才有理由训练条件化 Reader。

Confidence 是普通合法候选的分数校准。它改善了当前开发视频，不能因为使用共享 GMT 状态就赋予 JEV 创新名称。当前 MLP/Attention/CIC 仅有零残差原型、参数与工程等价/开销证据；没有训练、没有性能显著性，也不能声称优于它们。

所有当前控制器只调用相同原生匹配与验证流程，没有 JEV 专属二次全局求解。额外求解器因果贡献尚未构成新方法证据。历史其他阶段的 global re-solve 结果不能合并到当前完整 GMT 分数修正研究。

主要预测保留所有短轨，stock 结果另报。Rule 的额外人物 ID 与 IDSW 增加，说明只看 HOTA 或按短轨过滤后的结果会掩盖风险。身份持续量采用完整视频回顾性一对一对齐，是描述性相机帧量，不是真正在线已知的 owner 或秒级因果传播时间。

WWW 投稿证据仍缺：联合稳定增益、合格训练后普通控制器对照、独立完整视频、必要种子复现及可支持机制判断的消融。没有独立资格时，不读取封存数据或把开发视频当独立验证。目前为 NOT_ESTABLISHED。
''')
 if stock_temporal.exists():
  with (DOCS/'REVIEWER_ANALYSIS.md').open('a') as f:f.write('\n原生 stock 的长期身份诊断已补齐：Rule 回顾性持续量为 GMT 的0.7875倍，历史锚错误人物持续量为0.9313倍。最终轨迹长度过滤依赖整段视频，因此这仍不是在线干预的因果传播证明。缓存优化仅改变派生特征计算，先要求全视频逐项分数/特征与原预测严格一致，不能借此改策略或主评测协议。\n')
 if stock is not None:
  with (DOCS/'REVIEWER_ANALYSIS.md').open('a') as f:f.write('\n同约定原生 stock 基线已补齐：GMT HOTA68.389411/IDSW542；Rule HOTA68.966066/IDSW411/Frag608（GMT586）。其次要 HOTA/AssA/CVIDF1/IDSW/Frag 门槛通过，但不覆盖主要 raw 稳定性失败，stock 持续错误及效率仍待完整检查。Threshold 和 Confidence 的 stock IDSW 分别629、637，均超过1.05倍。原生最终短轨过滤与身份状态的交互可能影响结论，不能挑选协议，也不能简单称 Confidence 为全管线最佳。\n')
 resource=dict(new_retained_MiB=retained()/2**20,home_free_GiB=shutil.disk_usage('/home/liuyeqiang').free/2**30,GPU_wall_hours_including_live=spent())
 save(REPORTS/'ROUND_RESOURCE_SNAPSHOT.json',resource)
 report='''# GMT-JEV 中文研究报告（当前证据，效率任务仍以 JSON 终态为准）

## 1. 完整 GMT 基线

TRAIN16 单种子、1034 场景帧、2068 张真实图像；原始完整 Stage2 参数 53,515,309。HOTA 68.054089、AssA 67.064493、CVIDF1 86.544619、IDF1 87.182280、IDSW 1898、Frag 503、MOTA 87.000434。该视频为已暴露开发视频，与历史 DEV17–19 的 78.110 不可直接比较。

原项目源码冻结在 04324354；模型严格加载完整 Stage2，checkpoint SHA256 cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8。VFCE 1024、RPCE 128、GTA 1152。Original 与 Adapter OFF 的32帧和完整视频 logits、分数、提交、预测一致；零残差 CIC 的32帧原生等价也通过。参见 GMT_BASELINE_AUDIT.md 和 BASELINE_PARITY.json。

## 2. 可纠正错误机会

TRAIN12/13/14 冻结前缀共14541行；11305唯一认证行中25行不是当前合法目标，其中23行有另一摄像头历史。25行来自14个时间块；大量别名/短历史仍是 UNKNOWN。规则和单次有界校准探针均未产生认证接受纠错，未满足训练前置条件。初版拒绝接口的11次伪变化已有独立审计记录，旧结果保留。

## 3. 最小 CIC 输入与网络

post-GTA 24个观测特征：分数竞争、历史支持、年龄、相机内/间余弦、冲突与检测置信度。共享摘要仅来自已提交的 native ID。CIC 使用32维 edge/query/context 和两个观测冲突角色，4962参数；MLP5119、Attention4898。有界归一化残差上限0.05，乘原生 support count 加入 summed score，再由同一原生匹配/阈值处理。GT仅用于离线标签和评测。预注册 CE、preserve 与 residual loss 已定义，但没有执行训练，新增梯度更新为0。

## 4–5. HOTA 与身份指标

Threshold HOTA +0.135169，IDSW1898、Frag499；Rule HOTA +0.502582、AssA +0.836247、CVIDF1 +0.973228，但 IDSW2191（+15.44%）、Frag503；Confidence HOTA +0.261053、AssA +0.495980、CVIDF1 +0.376788、IDSW1839、Frag500。完整数表见 ONLINE_RESULTS.md。

Rule 未通过稳定性门槛；Confidence 尚未达到 +0.5。所有原始检测记录去除 track_id 后逐项一致。身份不一致持续量（回顾性相机帧）GMT5274、Threshold5090、Rule4680、Confidence5071；不能用持续量下降抵消 Rule 的 IDSW 与额外身份恶化。

## 6. 普通方法是否足以解释收益

目前所有正向结果来自未训练的普通控制器；学习式架构独立优势未建立。MLP/Attention/CIC 没有通过冻结 TRAIN 信号前置条件，不进行盲目训练或架构消融。不能把零残差原型的效率测量冒充训练后的公平质量比较。严格认证 N01/N10 为0/1、0/2、0/1，覆盖有限、Bank结果部分 UNKNOWN，不是完整闭环归因。

## 7. 真实效率

参见 EFFICIENCY.md 与 SPEED_METRICS.json。只报告同一 GPU7 上三次真实完整视频的已完成结果；未完成项 PENDING。计时包含原生最终后处理，场景/图像FPS单位另列。GPU4 的前缀矩阵操作审计单独记录，不用于 FPS。当前未测完，不能声明95%效率门槛通过。

## 8. WWW 科研资格

当前不能作为性能突破或 JEV 创新证明。缺联合质量优势、合格学习式对照、独立完整视频和种子复现。共享身份证据与跨相机条件化是待验证的机制假设，不能以命名替代对照。见 REVIEWER_ANALYSIS.md。

## 9. Goal 与下一步

应用 Goal 与仓库 MASTER_GOAL 均 ACTIVE，scientific_success=false。Rule 稳定方法 NO_GO，整个科学目标 NOT_ESTABLISHED。先完成已冻结的效率队列与 stock GMT 对照，不因开发正号追加阈值搜索；任何后续训练必须先有合格 TRAIN 信号。独立验证等待获准的未暴露范围，禁止新 DEV、sealed20–22、TEST、Full24。

## 10. 代码、证据与 GitHub

研究分支 research/gmt-jev-risk-association-20261014；[当前中文报告](https://github.com/LYQ1107/WWW/blob/research/gmt-jev-risk-association-20261014/docs/GMT_JEV/FINAL_RESEARCH_REPORT_ZH.md)。每阶段仅推源码、配置和紧凑JSON，权重、视频、图像、预测压缩包留在服务器并记录 SHA。实际发布终态由 runtime/publication/LAST_VERIFIED.json 的 local/remote HEAD 一致证明；报告生成来源 HEAD 与运行时源码 SHA 保存在 JSON。
'''
 if stock is not None:
  note='\n同约定原生最终输出：GMT HOTA68.389411、IDSW542、Frag586；Threshold HOTA68.541303、IDSW629、Frag582；Rule HOTA68.966066（+0.576655）、AssA69.235313、CVIDF188.912758、IDSW411（−24.17%）、Frag608（+3.75%）；Confidence HOTA68.642726、IDSW637、Frag568。Rule 次要指标出现正向，Threshold/Confidence 次要 IDSW 失败。stock 与 raw 均在结果前预注册，不能按结果换主要协议；stock 长期身份错误与效率待验证。\n'
  report=report.replace('## 6. 普通方法是否足以解释收益',note+'\n## 6. 普通方法是否足以解释收益')
  report=report.replace('stock 长期身份错误与效率待验证。','stock 长期身份诊断已补齐，效率仍待验证。')
 if stock_temporal.exists():
  note='\n原生 stock 持续量实测：回顾性 GMT4390、Rule3457（0.7875倍）；历史锚 GMT4046、Rule3768（0.9313倍）。主要 raw 结果仍保留，且全视频长度过滤使最终输出不构成在线因果证明。详细未知覆盖及删失语义见 STOCK_IDENTITY_TEMPORAL_AUDIT_ZH.md。可选归一化缓存保持当前所有控制器默认关闭，待全视频逐项等价审计后另测相同 Rule 的开销，不改变分数公式、权重或求解器。\n'
  report=report.replace('## 6. 普通方法是否足以解释收益',note+'\n## 6. 普通方法是否足以解释收益')
 if temporal.exists():
  d=read(temporal)
  text='\n新增结果后历史锚审计：仅使用更早场景已有至少3次已知观测、覆盖率>=80%的首个人物锚；屏蔽重复GT人物，未知/间断为右删失。认证错误人物关联持续相机帧'
  text+='、'.join(f"{k} {r['wrong_person_observed_duration_camera_frames']}" for k,r in d['methods'].items())
  text+='。Threshold 相比 GMT 增加6.64%，Confidence 减少17.46%；各方法最长区间均778帧。不能用未来整段多数身份映射重写历史归属，也不把此描述量称作干预因果传播时间。参见 PAST_ANCHOR_TEMPORAL_AUDIT_ZH.md；这是结果后补查，未改写预注册门槛。\n'
  report=report.replace('## 6. 普通方法是否足以解释收益',text+'\n## 6. 普通方法是否足以解释收益')
 if not pending:
  report=report.replace('（当前证据，效率任务仍以 JSON 终态为准）','（本轮完整证据）').replace('当前未测完，不能声明95%效率门槛通过。','全部三次计时已完成；各方法95%门槛以 SPEED_METRICS.json 的 efficiency_gate_pass 为准，质量与独立验证仍未通过。')
 report+='\n当前资源快照：'+json.dumps(resource,ensure_ascii=False)+'。只新增 home 文件，data1 读取历史资源；旧 NO_GO 与 checkpoint 未改写。\n'
 (DOCS/'FINAL_RESEARCH_REPORT_ZH.md').write_text(report)
 print('GMT_RESEARCH_REPORT_BUILT',speed['status'],pending,resource,flush=True)

if __name__=='__main__':main()

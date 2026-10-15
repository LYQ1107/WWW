"""Coherent current Chinese report from measured artifacts, no success selection."""
from gmt_jev_common import *

LABELS={'original':'GMT','threshold':'Threshold','rule':'Rule','confidence':'Confidence',
        'mlp_learned':'MLP','attention_learned':'Attention','cic_learned':'CIC'}

def main():
 training=read(REPORTS/'LEARNED_GATE_SMOKE_TRAINING.json')
 assert training['status']=='COMPLETE' and set(training['cases'])=={'mlp','attention','cic'}
 assert all(c['steps']==300 and c['seed']==SEED for c in training['cases'].values())
 base=read(REPORTS/'ORIGINAL_GMT_METRICS.json');b=base['metrics'];cases={'original':dict(primary_raw=base)}
 for kind in list(LABELS)[1:]:
  p=REPORTS/f'ONLINE_{kind}.json'
  if p.exists():cases[kind]=read(p)
 learned=read(REPORTS/'LEARNED_GATE_RESULTS.json')
 goal=read(REPORTS/'MASTER_GOAL.json');resources=read(REPORTS/'ROUND_RESOURCE_SNAPSHOT.json')
 lines=['# GMT-JEV 当前中文研究报告','',
  '当前结论：完整GMT恢复和集成等价已通过；固定300步普通MLP是已测开发质量正向候选。Attention/CIC未通过本轮质量门槛。已测质量、效率、机制、种子和独立资格分别判断，SCIENTIFIC_SUCCESS=false。',
  '本报告由当前JSON生成，所有方法均列出。主要协议固定为native raw；原生min_track_len=50及stock后处理另报。TRAIN16已暴露，所有结果属于开发证据。','',
  '## 1. 原始完整GMT基线','',
  '冻结原项目04324354源码和完整Stage2 checkpoint，严格加载411个tensor、53,515,309参数。VFCE1024+RPCE128形成1152维GTA输入，检测、GTA Transformer、原生候选/匹配、Gallery/Bank和REACT/WRITE全部保留。旧JEV executor、候选替代器和perception cache均关闭。',
  f"TRAIN16为1034场景帧、2068张真实图像；checkpoint SHA256：{STAGE2_SHA}。Original与Adapter OFF在32帧和完整1034场景的logits、分数、ID提交、Gallery/Bank及预测严格一致，见 BASELINE_PARITY.json。",
  '三个learned gate零残差头另有32场景的严格原生核验；该证据不冒充新模式的整段逐项零残差核验。旧零残差原型的完整stock指纹/效率结果单独保留。','',
  '## 2. 可纠正机会与监督资格','',
  '完整TRAIN12/13/14的冻结Rule主要HOTA均下降，合并ΔHOTA −0.008859，IDSW477→533；严格同状态接受式参考N01/N10=0/1。旧失败不改写，完整TRAIN结果不替代TRAIN16或独立验证。',
  '受限ORACLE是离线动作空间见证，未执行其动作，不是模型可表达性、最优上界或在线收益证明。手工门控排除了大部分认证错误目标，支持一次统一learned risk mode修订，不支持无限架构搜索。']
 pre_path=REPORTS/'LEARNED_GATE_SUPERVISION_PRECHECK.json'
 if pre_path.exists():
  p=read(pre_path);c=p['counts']['TRAIN']
  lines += ['',f"TRAIN时间桶共{c['rows']}行，其中{c['certified_unique_rows']}行有唯一认证合法目标；原GMT错误{c['E1_certified_original_wrong']}行，{c['supervised_E1_witness_rows']}行通过阈值margin及联合有界重评分见证。临时缓冲峰值{p['buffer']['peak_accounted_buffer_bytes']/2**20:.3f}MiB，保留全部{p['buffer']['retained_supervised_E1']}条合格E1和{p['buffer']['retained_supervised_preserve']}条preserve监督行。",
   'GT仅作离线认证；首个历史owner需至少3次已知观测且覆盖率≥80%，年轻别名也会阻止伪造唯一目标。UNKNOWN、目标冲突和有界不可达行不伪造监督，但完整候选矩阵仍参与原生约束训练。',
   '时间留出桶矩阵不进入optimizer，历史状态与认证跨桶连续；较早留出桶GT可参与随后TRAIN标签的历史认证。因此这不是严格标签隔离或独立验证。TRAIN16没有梯度更新。']
 lines+=['','## 3. 实际模块与训练','',
  '模块只在GTA聚合出native Global ID分数后添加残差：S_new=S_GMT+0.05×sigmoid(gate)×tanh(delta)×原支持长度。第一场景跨相机入口和后续普通MATCH均接入；原生候选、求解器、接受阈值和提交规则保留，Bank打分路径不替换。',
  '相同24维可观察输入涵盖GMT分数竞争、历史数量/年龄、同相机与其他相机外观支持、冲突、检测置信度和合法候选信息。派生摘要仅来自已提交的原生ID，不另建ID分配器或独立Gallery。',
  'MLP5119、Attention4898、CIC4962参数，CIC/Attention latent32，MLP hidden43。CIC增加观测冲突角色嵌入；不能因命名就声称JEV结构价值。',
  '三个头均已完成300步，使用相同临时样本、采样顺序、seed20261009、AdamW(lr0.001, weight_decay0.0001)、FP32及clip1；只保存固定300步终点，不按TRAIN16选step。原GMT冻结。Loss为native assignment结构化hinge+接受阈值hinge+10×preserve残差平方+全矩阵残差平方；margin0.005。真实样本梯度检查通过。具体顺序SHA、checkpoint SHA和训练记录见 TRAINING_METRICS.json / LEARNED_GATE_SMOKE_TRAINING.json。','',
  '## 4. 完整冷启动主要raw成绩','',
  '| 方法 | HOTA | ΔHOTA | AssA | DetA | IDF1 | CVIDF1 | CVMA | MOTA | IDSW | Frag |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for kind,r in cases.items():
  q=r['primary_raw'];m=q['metrics']
  lines.append(f"| {LABELS[kind]} | {m['HOTA']:.6f} | {m['HOTA']-b['HOTA']:+.6f} | {m['AssA']:.6f} | {m['DetA']:.6f} | {m['IDF1']:.6f} | {q['CVIDF1']:.6f} | {q['CVMA']:.6f} | {m['MOTA']:.6f} | {m['IDSW']:.0f} | {m['Frag']:.0f} |")
 lines+=['','MLP的已测raw质量门槛通过，IDSW仍增加2.48%，在预注册≤1.05倍范围内；MOTA下降约0.0718点，不能隐藏。Attention/CIC的HOTA、AssA、CVIDF1下降，CIC的IDSW还超过1.05倍，均未通过。以上为单种子开发差值，不是统计显著性或独立验证。','',
  '## 5. 身份稳定性与真实提交','',
  '| 头 | 正确接受式参考条数 | 保持率 | N01 | N10 | UNKNOWN目标 | 回顾性不一致相机帧 | 历史锚错误相机帧 |',
  '|---|---:|---:|---:|---:|---:|---:|---:|']
 for kind in ['mlp','attention','cic']:
  if kind not in learned['methods']:continue
  r=learned['methods'][kind];c=r['paired_current_state_counts'];raw=cases[kind+'_learned']
  lines.append(f"| {kind} | {c.get('reference_correct',0)} | {100*r['correct_accepted_reference_preservation_rate']:.6f}% | {c.get('N01',0)} | {c.get('N10',0)} | {c.get('UNKNOWN_target',0)} | {raw['identity']['identity_inconsistency_duration_camera_frames']} | {r['post_hoc_past_anchor_temporal']['wrong_person_observed_duration_camera_frames']} |")
 lines+=['','GMT的回顾性不一致持续量为5274相机帧，严格历史锚错误持续量为4054。MLP两者分别为4407和3795。前者依赖整段映射，后者只用更早场景的首个≥3/80%认证owner，并对未知/缺失/视频结束作删失；两者都不是真实单次干预的因果传播秒数。',
  'N01/N10只与同一当前变异状态下原始GMT的接受式MATCH比较，原始拒绝后的Bank反事实未知；不同方法的认证覆盖不同。HOTA正号不能自动解释为这些局部计数的正向。']
 pair=REPORTS/'LEARNED_FIRST_DIVERGENCE_NATIVE.json'
 if pair.exists():
  p=read(pair)
  lines+=['','首次可见ID分歧在frame2/view1/row22。三场景重跑确认，事件前完整native状态/RNG、派生摘要、原始分数、24维输入、候选和支持长度严格一致；GMT实际创建ID52，MLP接受已有ID25。两条前缀与各自完整轨迹一致，证明实际提交有变更。',
   '但该行GT为UNKNOWN，不能计为认证纠错。未执行事件单独开启/关闭的完整视频干预，不能将全视频增益归因于此，也未证明真正错误传播时间。见 LEARNED_FIRST_DIVERGENCE_NATIVE.json。']
 lines+=['','## 6. 普通对照与结构价值','',
  '三头使用相同证据、合法候选、原生求解器、残差上界、数据和训练预算。当前普通MLP优于Attention/CIC，按任务要求保留MLP，不为JEV强行扩大架构。CIC当前fixed300开发方案NO_GO，整个研究目标仍未完成。',
  '这不证明跨相机特征必需，也不排除普通分数/阈值校准解释大部分收益。缺少训练后证据消融、合格种子和独立视频，不能宣称JEV-inspired结构优于普通控制器或已满足WWW创新要求。所有当前修正仍使用同一native solver，历史额外global re-solve结果不并入本研究。','',
  '## 7. 真实完整图像效率','',
  '| 方法 | 完整图像FPS | 相对GMT | 三轮状态 | 95%门槛 |','|---|---:|---:|---|---|']
 baseline_speed=read(REPORTS/'SPEED_original.json')
 for kind in LABELS:
  p=REPORTS/f'SPEED_{kind}.json'
  if p.exists():
   s=read(p);assert len(s['trials'])==3;ratio=s['mean_full_scene_FPS']/baseline_speed['mean_full_scene_FPS']
   lines.append(f"| {LABELS[kind]} | {2*s['mean_full_scene_FPS']:.6f} | {ratio:.6f} | COMPLETE | {ratio>=.95} |")
  else:
   p=OUT/'speed'/kind/'PROGRESS.json';n=len(read(p)['trials']) if p.exists() else 0
   lines.append(f'| {LABELS[kind]} | PENDING | PENDING | {n}/3 | PENDING |')
 cache=learned['methods'].get('mlp',{}).get('cached_deployment')
 cache_job=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs'].get('P5_mlp_learned_cached_full_GPU7')
 cache_state=cache_job['status'] if cache_job else 'NOT_REGISTERED'
 proof_path=REPORTS/'LEARNED_MLP_FEATURE_CACHE_PARITY.json'
 cache_proof=read(proof_path)['status'] if proof_path.exists() else 'PENDING'
 cache_description=f'缓存工程核验：{cache_proof}；计时注册状态：{cache_state}。'
 if cache:
  cache_description+='完整1034场景、2067次features/gate/residual/score及raw预测严格一致，三轮stock指纹与默认MLP相同，复用同一质量成绩。'
  cache_description+=f"三轮缓存速度门槛为{cache['efficiency_gate_pass']}；默认未缓存结果仍单独保留。"
 else:
  cache_description+='尚无已完成的三轮缓存均值，不给出效率通过结论。'
 cache_description+='协议见 LEARNED_MLP_CACHE_PROTOCOL.md；旧Rule缓存负向结果保留。'
 if cache:lines.append(f"| MLP缓存（单列） | {cache['full_image_FPS']:.6f} | {cache['FPS_ratio_to_GMT']:.6f} | COMPLETE | {cache['efficiency_gate_pass']} |")
 lines+=['','同一GPU7/V100-DGXS-32GB，每次重置原生状态、派生摘要和seed。包含真实图像读取/resize、检测、VFCE/RPCE、GTA、原生匹配/Gallery/Bank及最终stock处理；不含模型加载、8帧warmup和GT评测。图像FPS=2×场景FPS。Stage2 p50/p95和峰值显存见各 SPEED_*.json。共享宿主机其他卡负载存在变化，串行块不能独立分离系统负载、状态轨迹和模块开销的因果贡献。',
  'MLP默认平均完整图像FPS为3.129765、相对GMT0.917882，未通过95%，不能把质量正向称为完整目标成功。24场景GPU8组件诊断只有定位意义，不外推全视频FPS。',
  cache_description,'',
  '## 8. 同约定stock次要结果','',
  '| 方法 | HOTA | AssA | CVIDF1 | IDSW | Frag |','|---|---:|---:|---:|---:|---:|']
 stock_path=REPORTS/'ORIGINAL_GMT_STOCK_METRICS.json'
 if stock_path.exists():
  stock={'original':read(stock_path)};stock.update({k:r['secondary_stock'] for k,r in cases.items() if k!='original'})
  for kind,q in stock.items():
   m=q['metrics'];lines.append(f"| {LABELS[kind]} | {m['HOTA']:.6f} | {m['AssA']:.6f} | {q['CVIDF1']:.6f} | {m['IDSW']:.0f} | {m['Frag']:.0f} |")
 lines+=['','stock不是主要协议。最终轨迹长度过滤依赖整段视频，不能用其较好结果覆盖raw失败或称作在线已知的因果修复。官方MATLAB toolkit SHA不变，TrackEval/转换器协议冻结；具体原始输出和输入SHA见各ONLINE报告。','',
  '## 9. Goal、资格和下一步','',
  f"MASTER_GOAL状态 {goal['status']}，scientific_success={goal['scientific_success']}。完整基线和工程接入已证；MLP已测质量正向，默认效率失败；必要种子、MCMT特征/机制贡献和合格独立完整视频尚未证明。应用Goal不能据此标为科学完成。",
  '数据范围仅TRAIN12/13/14学习、TRAIN16开发；不新增DEV17–19、封存20–22、官方TEST、Full24或新数据下载。TRAIN16不能伪装独立验证，旧阶段NO_GO/checkpoint保留。',
  ('默认计时、完整缓存等价和三轮缓存计时均已完成。本轮不扩大架构/训练步数；效率失败时保留质量候选并报告精度与速度权衡，不能宣布完整目标成功。即使效率满足，仍需必要训练种子和合法合格独立验证。' if cache else
   '完成当前已注册缓存计时后统一结算，不扩大架构/训练步数。未完成三轮或未通过95%就报告效率尚未通过；必要训练种子、机制和合法合格独立验证仍待补足。'),
  '最近资源快照（并非停止时最终结算）：'+json.dumps(resources,ensure_ascii=False)+'。累计上限12 GPU-wall-hours、新留存512MiB、目标盘至少30GiB余量，重设Goal不清零预算。','',
  '## 10. 代码、记录与GitHub','',
  '[研究分支当前报告](https://github.com/LYQ1107/WWW/blob/research/gmt-jev-risk-association-20261014/docs/GMT_JEV/FINAL_RESEARCH_REPORT_ZH.md)。原始完整GTA checkpoint、视频、图像、预测压缩包和训练权重留服务器，仅推必要源码和紧凑记录；各自SHA已记录。',
  '运行起点HEAD和源码SHA见 EXPERIMENT_REGISTRY.json，具体结果含source binding。发布是否成功以runtime/publication/LAST_VERIFIED.json的local/remote HEAD一致为准；报告生成HEAD不是未来发布提交号。',
  '关键证据：BASELINE_PARITY.json、FULL_TRAIN_MATCHED_QUALITY_AUDIT.json、LEARNED_GATE_SUPERVISION_PRECHECK.json、LEARNED_GATE_SMOKE_TRAINING.json、LEARNED_GATE_RESULTS.json、LEARNED_FIRST_DIVERGENCE_NATIVE.json、SPEED_METRICS.json、GO_NO_GO.json。']
 forecast=REPORTS/'CACHE_TIMING_RESOURCE_REASSESSMENT.json'
 if forecast.exists():
  f=read(forecast);lines+=['',f"缓存计时资源复核（计时前记录）：原固定0.7小时估计的停止记录保留；按相关完整作业实测墙钟耗时×1.10+10秒，估计{f['estimated_seconds']:.3f}秒，复核时剩余{f['remaining_GPU_seconds']:.3f}秒，结论{f['status']}。没有改变12小时硬上限、模型、数据或三轮协议；实际未完成三轮就不能通过效率门槛。见 CACHE_TIMING_RESOURCE_REASSESSMENT.json。"]
 from gmt_jev_tradeoff_figure import main as plot_tradeoff
 plot_tradeoff()
 lines+=['','![已完成测量的开发精度与效率](GMT_JEV_ACCURACY_EFFICIENCY.png)',
  '图只列已完成三轮计时的方法，保留旧Rule缓存负向结果。横误差线为实测三轮范围，不是置信区间；阴影仅表示HOTA/FPS数值目标，不能替代身份、机制、种子和独立资格。PDF和源数据见同目录及 ACCURACY_EFFICIENCY_FIGURE.json。']
 (DOCS/'FINAL_RESEARCH_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
 print('CURRENT_CHINESE_REPORT_REBUILT',len(cases),'methods',flush=True)

if __name__=='__main__':main()

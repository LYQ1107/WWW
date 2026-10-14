"""Summarize frozen development controls; no model/data selection or job launch."""
import gzip
from gmt_jev_common import *

def predictions(r):
 with gzip.open(r['path'],'rt') as f:return json.load(f)

def main():
 baseline=read(REPORTS/'ORIGINAL_GMT_METRICS.json')
 base=baseline['metrics'];base['CVIDF1']=baseline['CVIDF1']
 raw=predictions(baseline['predictions'])
 geometry=lambda rows:[{k:v for k,v in r.items() if k!='track_id'} for r in rows]
 arms={};lines=['| 方法 | HOTA | ΔHOTA | AssA | CVIDF1 | IDF1 | IDSW | Frag | MOTA |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 def row(name,m):
  return '| '+name+' | '+' | '.join(f'{m[k]:.3f}' for k in ['HOTA'])+f" | {m['HOTA']-base['HOTA']:+.3f} | "+' | '.join(f'{m[k]:.3f}' for k in ['AssA','CVIDF1','IDF1'])+f" | {m['IDSW']:.0f} | {m['Frag']:.0f} | {m['MOTA']:.3f} |"
 lines.append(row('Original GMT',base))
 for kind in ['threshold','rule','confidence']:
  r=read(REPORTS/f'ONLINE_{kind}.json');m=dict(r['primary_raw']['metrics'],CVIDF1=r['primary_raw']['CVIDF1'])
  same=geometry(raw)==geometry(predictions(r['raw_predictions']));assert same,kind
  duration_ratio=r['identity']['identity_inconsistency_duration_camera_frames']/baseline['identity']['identity_inconsistency_duration_camera_frames']
  gates=dict(HOTA=m['HOTA']>=base['HOTA']+.5,AssA=m['AssA']>=base['AssA'],CVIDF1=m['CVIDF1']>=base['CVIDF1'],
   IDSW=m['IDSW']<=base['IDSW']*1.05,Frag=m['Frag']<=base['Frag']*1.05,identity_duration=duration_ratio<=1.05)
  arms[kind]=dict(primary_raw=m,secondary_stock=r['secondary_stock'],delta={k:m[k]-base[k] for k in m},
   development_quality_gates=gates,all_quality_gates_pass=all(gates.values()),perception_geometry_and_scores_exact=same,
   identity=r['identity'],identity_duration_ratio=duration_ratio,counts=dict(N01=r['counts'].get('N01',0),N10=r['counts'].get('N10',0),**{k:v for k,v in r['counts'].items() if k not in ['N01','N10']}),
   report=ref(REPORTS/f'ONLINE_{kind}.json'),FPS='PENDING_SEPARATE_SAME_GPU_FULL_IMAGE_MEASUREMENT',independent_validation=False)
  lines.append(row(kind,m))
 result=dict(status='COMPLETE_DEVELOPMENT_CONTROLS',video=16,frames=1034,views=2,seed=SEED,baseline=dict(primary_raw=base,identity=baseline['identity'],report=ref(REPORTS/'ORIGINAL_GMT_METRICS.json')),
  methods=arms,primary_protocol='UNFILTERED_NATIVE_ONLINE',secondary_protocol='UNCHANGED_NATIVE_STOCK_MIN_TRACK_LEN_50_AND_POSTPROCESS',
  matched_original_stock_baseline='PENDING_SPEED_ORIGINAL_FIRST_TRIAL_PREDICTIONS',
  new_gradient_updates=0,learned_architecture_advantage='NOT_ESTABLISHED',scientific_success=False,
  caution='TRAIN16 is exposed development only. Row N01/N10 are partial same-current-state diagnostics with rejected-base Bank outcomes UNKNOWN, not complete-video causal attribution.',source=binding())
 save(REPORTS/'ONLINE_METRICS.json',result)
 (DOCS/'ONLINE_RESULTS.md').write_text('''# 完整原生在线对照：开发结果

固定 TRAIN16：1034 个场景时刻、2068 张真实图像、单种子。完整 GMT Stage2、RPCE/GTA、合法候选、求解器、Bank/Gallery、REACT/WRITE 均保留，只有 post-GTA 分数修正不同。三个方法从相同冷启动运行真实 mutated-state 闭环，没有新增训练更新。

主要评测是预注册的未过滤原生预测，采用相同 TrackEval 与 MATLAB 跨摄像头工具链。

'''+ '\n'.join(lines)+'''

Rule 的 HOTA +0.502582，但 IDSW 2191/1898=1.1544，超过允许的 1.05 倍；其 MOTA 也下降。因此不能以 HOTA 单项正向认定成功。Confidence 的 HOTA +0.261053、IDSW −59，但未达到 +0.5 的开发目标。Threshold 的 HOTA +0.135169，同样未达到目标。三个方法的 AssA/CVIDF1 均上升。

身份不一致持续量（相机帧，回顾性一对一 ID/person 对齐）分别为 GMT 5274、Threshold 5090、Rule 4680、Confidence 5071；最长连续段均为 464。这是描述性统计，不能当作在线真身份认证或干预的完整因果归因。额外人物 ID 数分别为 837、825、1160、737；Rule 的碎片/出生问题不能只看 Frag 或总持续量。

三条干预与 GMT 的全部 58467 条原始检测记录，在去除 track_id 后逐项严格相同，确认前端几何、检测分数和检测集合一致。HOTA DetA 的小幅变化不能解释成新增检测；它的匹配也受到身份对齐影响。

同当前状态的严格认证行 N01/N10：Threshold 0/1、Rule 0/2、Confidence 0/1。大量原始拒绝后的 Bank 结果及多别名目标为 UNKNOWN；这些局部计数不是完整闭环纠错率，也不能推翻或替代真实视频指标。

原生 min_track_len=50 与 stock 后处理的结果保存在每条 ONLINE_*.json 的 secondary_stock；原始 GMT 的同约定基线尚待计时任务保留的输出评测，当前不计算 stock 差值、不混用 raw 基线。

完整图像 FPS 在空闲同一 GPU 的独立任务中测量，质量诊断运行的秒数不充当 FPS。没有训练 MLP/Attention/CIC：冻结 TRAIN12/13/14 的规则和有界探针均无认证接受纠错信号，未满足预注册训练前置条件。不能因 TRAIN16 的正号转而调整网络或训练样本。

当前结论：普通控制器能够产生开发集指标变化；联合质量目标没有通过。独立视频、种子复现、学习式普通控制器优势及 WWW 架构创新均未建立。
''')
 (DOCS/'ERROR_ANALYSIS_ZH.md').write_text('''# GMT-JEV 错误机会与闭环风险

冻结 TRAIN12/13/14 各前 256 帧的 14541 行中，11305 行有唯一合法历史认证目标，11280 行 GMT 已正确，25 行不同于该目标。25 行中 23 行存在另一摄像头历史。多别名 2675 行、GT UNKNOWN 224 行、短历史 149 行、无唯一目标 148 行、不活跃唯一 ID 40 行均保留为不可监督或非活跃诊断；不能把它们补成负例。25 行聚集在 14 个时间块，不是 25 个独立样本，也不是全系统错误总量或 HOTA 上限。

跨相机冲突规则在同状态没有认证接受的纠错或反改错；单次有界 confidence/first-link 探针也没有。初版把拒绝的 −1 与实际新生 ID 比较得到 11 次变化，接口审计已记录并修正，旧诊断仍保留，未据此启动训练。

完整 TRAIN16 的规则修正使 HOTA 增加 0.503，同时新生 ID 1695、IDSW 2191、额外人物 ID 1160，稳定性门槛失败。局部前缀的零认证信号与完整开发视频 HOTA 上升并不矛盾：许多改变发生在拒绝/Bank/别名 UNKNOWN 行，状态反馈会改变后续候选与身份轨迹。当前证据不足以把全视频收益归因于正确的跨相机动作判断。

Confidence 只压低原先低于接受阈值的分数；它可以改变原生 Hungarian 竞争，再经完全相同的原生阈值校验，并非不影响在线分配。该普通校准获得较小正向结果，必须作为正式传统控制器记录，不能称为学习式 JEV 的优势。

下一步先完成相同 GPU 的实测效率和 matched stock 指标，保留所有正负结果。没有合格训练信号时，不扩大网络，不以开发视频挑参数，不读取封存集或 TEST；科学 Goal 保持 ACTIVE/NOT_ESTABLISHED。
''')
 goal=read(REPORTS/'MASTER_GOAL.json');goal['gates'].update(MATCHED_DEVELOPMENT_GAIN='RULE_HOTA_PLUS_0_502582_BUT_JOINT_TARGET_FAIL',IDENTITY_STABILITY='RULE_FAIL_IDSW_1_1544X',ORDINARY_CONTROL_ADVANTAGE='NOT_ESTABLISHED_NO_LEARNED_TRAINING',SCIENTIFIC_SUCCESS='NOT_ESTABLISHED')
 goal['scientific_success']=False;goal['status']='ACTIVE';save(REPORTS/'MASTER_GOAL.json',goal)

if __name__=='__main__':main()

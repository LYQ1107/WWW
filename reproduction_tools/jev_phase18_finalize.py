"""Finalize bounded evidence without completing the scientific goal."""
from jev_phase18_common import *
def main():
 protect();out=read(REPORTS/'NATIVE_PAIRED_OWNER_OUTCOMES.json');s=out['summary']
 assert s['total_complete_native_branches']==73
 gate=dict(status='NO_GO',qualified=False,ENGINEERING='PASS_NATIVE_FORK_SCORE_STATE_PARITY',strict_CLEAR='PARTIAL_DUPLICATE_GT_VIDEO16',DATA_ELIGIBILITY='INSUFFICIENT_INDEPENDENT_SAFE_TRAIN_AND_RESERVED',SCIENTIFIC='NO_GO_LEARNING',DEPLOYMENT='NOT_EVALUATED',INDEPENDENT_VALUE='NOT_ESTABLISHED',summary=s,main_hypothesis='Full census identifies additional opportunities; independent safe supervision remains unqualified',preserves_prior_failures=True,no_new_training=True,no_new_policy_metrics=True,sealed_20_21_22=True,official_TEST=False,Full24=False)
 save(REPORTS/'LEARNING_ELIGIBILITY.json',gate)
 save(REPORTS/'FINAL_GO_NO_GO.json',dict(gate,master_goal_complete=False,next_scope_authorized='user approved TRAIN02–09 / RESERVED10,11,15 audit and bounded experiments; no training',extra_scope_proposal='docs/PHASE18_NEXT_SCOPE_PROPOSAL.md'))
 table='\n'.join('|'+ '|'.join(map(str,[g['selection_index'],g['event_key'],g['split'],g['category'],','.join(g['safe_variants']) or '—',','.join(g['harmful_variants']) or '—']))+'|' for g in out['groups'])
 counts=read(REPORTS/'COMPLETE_NATIVE_OWNER_CENSUS.json')['per_video_counts']
 ct='\n'.join(f'|{v}|{counts[str(v)]["events"]}|{counts[str(v)].get("AVAILABLE_ERROR",0)}|{counts[str(v)].get("UNAVAILABLE_ERROR",0)}|' for v in TRAIN+DIAGNOSTIC)
 report=f'''# Phase XVIII — Native owner / candidate evidence report

本轮真实执行完成，学习资格仍 NO_GO；MASTER 科研目标没有完成。冻结 Phase XVII d44c30d 与旧负结果，只增加新代码与诊断。用户已确认下一步追加 TRAIN02–09、预留10/11/15的数据审计与有界实验；本轮输出不作为追加数据上的结果。

## 完整轨迹揭示什么

严格授权 TRAIN12/13/16 共90,058个观测，5,598个owner错误/假拆分且正确owner在合法active MATCH池中的事件。14单列旧重复GT诊断。“严格授权TRAIN”指划分，不代表video16全部GT唯一。事件多并不等于独立、可安全学习的纠错标签。

|video|观测|AVAILABLE_ERROR|UNAVAILABLE_ERROR|
|---|---:|---:|---:|
{ct}

“UNAVAILABLE”只指active MATCH；接受MATCH时并未构造反事实Bank候选，不能推成所有历史都不存在正确身份。当前纯净支持与最早不可变owner分别统计；owner由过去单GT≥3、覆盖≥0.8建立，污染后仍保留，不是完美GT归属证明。GT不进入actor。

完整余弦排名、正确候选覆盖、混合历史与NN独立行偏好见FULL_CENSUS_EVIDENCE_SEPARABILITY.json。video16有5,247个可用错误，其中4,905个所有正确anchored-owner候选历史均已混合，GT-free备选仅覆盖3,678个；不能以全池Recall接近100%推断可可靠判断owner。该审计是描述性证据，不证明信息论不可能，也不是学习策略表现。

## 实际干预与安全归因

64个时间冲突组固定采样；16组执行16基线+57备选，共73个完整原生H64分支，同时检查H16/32/64。每分支从相同完整native状态和RNG恢复，第一步候选/分数/合法性完全一致；BASE续帧ID与完整原轨迹一致，原轨迹恢复后IDs、向量与raw memory哈希逐步一致。替代动作只依事前NN/余弦规则，不按GT选择。

冻结XV native接口实际只有一次MATCH Hungarian，然后残余Bank/REACT、NEW和Gallery写入。新探针锁定一个合法ref或private DEFER，再用同一个lawful_choice求解余行并检查容量。没有第二轮MATCH评分校验；此处是因果动作探针，不是已经可部署的学习controller。

|组|event key|划分|类别|严格安全纠错变体|有害变体|
|---:|---|---|---|---|---|
{table}

共有{s['real_correction_action_variants']}个动作含真实纠错，{s['harmful_action_variants']}个动作存在至少一项有害后果；两者可重叠。{s['safe_action_variants']}个动作、{s['safe_temporal_representatives']}个时间代表通过严格局部安全；owner-only安全诊断{s['owner_only_safe_action_variants']}个不能替代严格CLEAR资格。纠错必须是已知错误→已知正确，或有事前anchored支持的目标假拆分恢复；错误→UNKNOWN不算纠错。逐行N01/N10、UNKNOWN、污染/拆分、其他原MATCH行重配、每GT-camera错误传播和CLEAR差异见NATIVE_PAIRED_OWNER_OUTCOMES.json。

安全判定同时检查所有受影响行、三个H：反改错、正确覆盖丢失、新混合/假拆分、更多错误观测、任一GT-camera更长错误段、更多严格IDSW均否决。错误传播按真实检测观测计数，缺帧与不可见区间删失，不能声称完整不可见期间已知错误持续时间。

## 为何仍不训练

在查看主干预结果前冻结的跨时间owner/conflict复核，将16组并为6个保守单位；2个单位同时含TRAIN与RESERVED，不能贡献独立预留验证。结果后未重采样。真正合格安全TRAIN={s['independent_safe_TRAIN_groups']}、RESERVED={s['independent_safe_RESERVED_groups']}，少于原预注册8+3。邻帧、相关备选及三个重叠H均不是独立样本，没有显著性/泛化声明。

video16首轮严格TrackEval在View2 timestep184发现重复GT ID44而中止。12/13成功结果保留，16已完成的两组未重复，余四组按相同状态/动作恢复执行。受重复GT影响的过去前缀或未来窗口，CLEAR置null；不删除/重编号GT，也未调用permissive scorer。{s['strict_CLEAR_unevaluable_action_variants']}个动作无法获得全部H的严格CLEAR资格。原始失败队列、日志、部分结果和修复actor均保留，工程失败不解释成算法正负收益。

## 结论与下一步

冻结困难前缀确实未覆盖完整轨迹中大量候选可用错误，且部分原生合法动作能安全纠错；但许多局部纠错产生联合关联副作用，监督独立性和安全性仍不合格。当前结果没有证明JEV结构优于同证据MLP/Attention，更没有证明未训练的新架构可泛化。

新训练、同证据学习基线、三生命周期新共享模型、完整冷启动新policy、部署速度和独立controller测试全部NOT_RUN；新HOTA/AssA/IDF1/MOTA为null。既有Phase XV/XVII反复DEV指标不是本轮独立结果。20/21/22、官方TEST、Full24均未启动。下一方案和8GiB资源上限见PHASE18_NEXT_SCOPE_PROPOSAL.md，追加范围先做历史暴露/GT/缓存审计，失败不换有利序列。

代码：jev_phase18_census/select/independence/contract/intervene/metrics/outcomes/separability/finalize及native constraint；持续研究controller已支持Phase XVIII。实际actor版本、冻结checkpoint SHA、输入/输出哈希由各JSON绑定。GitHub只推紧凑代码、报告、统计与manifest；视频、权重、原生状态和逐帧日志保留服务器。
'''
 (ROOT/'docs/PHASE18_FINAL_RESEARCH_REPORT.md').write_text(report)
 master=read(MASTER/'MASTER_GOAL.json')
 master.update(master_status='ACTIVE',current_phase_status='XVIII_COMPLETE_PIVOT_TO_AUTHORIZED_SCOPE_AUDIT',next_experiment='E19_SCOPE_AUDIT',paper_readiness='NOT_READY',scientific_SUCCESS=False,extra_scope_authorization=dict(TRAIN=list(range(2,10)),RESERVED=[10,11,15],source='user asynchronous answer: 按该方案追加数据审计与有界实验',training=False,runtime_GiB=8))
 save(MASTER/'MASTER_GOAL.json',master)
 registry=read(MASTER/'EXPERIMENT_REGISTRY.json')
 for e in registry['experiments']:
  if e['id']=='E18_INTERVENE':e.update(status='COMPLETE',evidence=[ref(REPORTS/'NATIVE_PAIRED_OWNER_OUTCOMES.json')])
  if e['id']=='E18_QUALIFY':e.update(status='NO_GO',evidence=[ref(REPORTS/'LEARNING_ELIGIBILITY.json')],reason='No independent qualified8+3; no training')
 save(MASTER/'EXPERIMENT_REGISTRY.json',registry)
 save(MASTER/'NEXT_ACTION.json',dict(status='AUTHORIZED_SCOPE_AUDIT_NEXT',experiment='E19_SCOPE_AUDIT',TRAIN=list(range(2,10)),RESERVED=[10,11,15],no_training=True,source_proposal=str(ROOT/'docs/PHASE18_NEXT_SCOPE_PROPOSAL.md'),scientific_success=False))
 evidence=read(MASTER/'EVIDENCE_REGISTRY.json')
 evidence['entries'] += [ref(p) for p in sorted(REPORTS.glob('*.json')) if str(p) not in {e['path'] for e in evidence['entries']}]
 save(MASTER/'EVIDENCE_REGISTRY.json',evidence)
 with (MASTER/'RESEARCH_DECISIONS.md').open('a') as f:f.write(f'\nXVIII completed73 native branches. Independent safe TRAIN/RESERVED={s["independent_safe_TRAIN_groups"]}/{s["independent_safe_RESERVED_groups"]}; learning NO_GO. Duplicate-GT strict failure retained, affected CLEAR null; no annotation changes. User approved additional TRAIN02–09 / RESERVED10,11,15 audit and bounded experiments, <=8GiB, no training. Master remains ACTIVE; scientific success false.\n')
 (ROOT/'docs/WWW_JEV_RESEARCH_ROADMAP.md').write_text('''# Research roadmap

XVII冻结负结果与有界机制诊断。XVIII完整普查、候选/owner审计与73条真实原生分支完成，独立安全监督未通过，训练NOT_RUN。MASTER持续ACTIVE，科研与论文NOT_READY。

用户已授权追加TRAIN02–09、预留10/11/15；先审计历史暴露、GT重复与缓存，冻结资源与独立视频分割，再运行有界实验，最多64组、16组干预、80分支。细节见PHASE18_NEXT_SCOPE_PROPOSAL.md。不得解封20/21/22，不读取官方TEST，不Full24。旧Stage1全24暴露与旧controller暴露必须披露。

只有合格监督、反改错负例、原生安全与独立预留资格齐备，才冻结轻量候选模型、同证据MLP/Attention，再考虑三生命周期共享、完整在线性能、身份索引检索和独立controller验证。没有资格不训练；相关动作/窗口不作为独立样本。
''')
 with (ROOT/'docs/WWW_JEV_PAPER_READINESS.md').open('a') as f:f.write('\nXVIII完成完整TRAIN owner普查与73条原生未来分支，独立安全监督仍NO_GO；重复GT使部分CLEAR不可评估。未建立新学习策略、架构增量或独立泛化，论文仍NOT_READY。追加数据仅进入已授权审计与有界诊断，不代表训练资格。\n')
 print('PHASE18_FINALIZED',s)
if __name__=='__main__':main()


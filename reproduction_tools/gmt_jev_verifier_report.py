"""Truthful V2 evidence report; conditional experiments stay explicitly unrun."""
from gmt_jev_verifier_common import *
from gmt_jev_verifier_job import spent

def optional(name):
    p=REPORTS/(name+'.json');return read(p) if p.exists() else None
def write(name,lines):(DOCS/name).write_text('\n'.join(lines)+'\n')

def main():
    audit=optional('PAIRED_PLAN_OPPORTUNITY');q1=optional('Q1_RELIABILITY_DIAGNOSTIC')
    closure=optional('NATIVE_PAIRED_PLAN_OPPORTUNITY')
    goal=read(REPORTS/'GOAL.json');freeze=read(REPORTS/'BASELINE_FREEZE.json')
    complete=audit is not None and closure is not None;eligible=bool(complete and closure['Q2_training_eligible'])
    decision='BLOCKED_INSUFFICIENT_SUPERVISION' if complete and not eligible else 'CONTINUE_TYPED_JEV'
    if not complete:decision='P0_NATIVE_CLOSURE_RUNNING' if audit else 'P0_RUNNING'
    counts=audit['counts'] if audit else {};support=closure['Q2_support'] if closure else audit['Q2_support'] if audit else {}
    base=legacy.read(legacy.REPORTS/'ORIGINAL_GMT_METRICS.json')
    mlp=legacy.read(legacy.REPORTS/'ONLINE_mlp_learned.json')
    annotation=ref(ANNOTATIONS)
    assert annotation['SHA256']==legacy.read(legacy.REPORTS/'TRAIN_MATCHED_original_VIDEO12.json')['inference']['annotation']['SHA256']
    live=[n for n,j in read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs'].items() if j['status']=='RUNNING']
    resources=dict(created_Asia_Shanghai=now(),new_GPU_hours_including_live=spent(),new_GPU_hours_max=4,
        old_GPU_hours_unchanged=read(REPORTS/'ROUND_BASELINE.json')['V1_GPU_hours'],new_retained_MiB=new_bytes()/2**20,
        home_free_GiB=shutil.disk_usage(ROOT).free/2**30,registered_live_jobs=live)
    save(REPORTS/'RESOURCE_ACCOUNTING.json',resources)
    goal['gates']['GMT_BASELINE_PRESERVED']='PASS_EXACT_ALL_APPROVED_TRAIN_RAW_RECORDS' if audit else 'FROZEN_FILES_AND_PRIOR_EXACT_PARITY'
    goal['gates']['Q1_SUPERVISION_ELIGIBLE']='SEE_Q1_CLASS_AND_ID_SUPPORT' if audit else 'AUDIT_RUNNING'
    goal['gates']['Q2_SUPERVISION_ELIGIBLE']='PASS_MINIMUM_ENGINEERING_GATE' if eligible else 'FAIL_INSUFFICIENT_CERTIFIED_PLAN_SUPPORT' if complete else 'AUDIT_RUNNING'
    if complete and not eligible:
        goal['status']='BLOCKED_INSUFFICIENT_SUPERVISION'
        for key in ['TYPED_DECISION_CONNECTED','NATIVE_ACTION_PARITY','DEVELOPMENT_HOTA_GAIN','IDENTITY_STABILITY','ORDINARY_VERIFIER_COMPARISON','JEV_STRUCTURE_ABLATION','FULL_IMAGE_FPS']:
            goal['gates'][key]='NOT_RUN_Q2_SUPERVISION_GATE_FAILED'
        goal['gates']['NATIVE_ACTION_PARITY']='PASS_LOCAL_NATIVE_TWIN_PLAN_AND_ROLLBACK_CONTRACTS_NOT_DEPLOYED_VERIFIER_OFF'
    goal['gates']['SCIENTIFIC_SUCCESS']='NOT_ESTABLISHED';goal['scientific_success']=False
    save(REPORTS/'GOAL.json',goal)
    pointer=read(ROOT/'reports/CURRENT_RESEARCH_GOAL.json')
    assert pointer['master_goal']=='reports/GMT_JEV_VERIFIER/GOAL.json'
    pointer['status']=goal['status'];save(ROOT/'reports/CURRENT_RESEARCH_GOAL.json',pointer)
    current=ROOT/'CURRENT_RESEARCH_GOAL.md'
    text=current.read_text()
    for state in ['ACTIVE','BLOCKED_INSUFFICIENT_SUPERVISION']:
        text=text.replace('V2状态 '+state,'V2状态 '+goal['status'])
    current.write_text(text)
    final_goal=DOCS/'FINAL_GOAL.md';text=final_goal.read_text()
    for state in ['ACTIVE','BLOCKED_INSUFFICIENT_SUPERVISION']:
        text=text.replace('状态 '+state+'；','状态 '+goal['status']+'；')
    final_goal.write_text(text)
    training=dict(status=q1['status'] if q1 else 'PENDING_Q1_OR_ELIGIBILITY',Q1_diagnostic=q1,
        typed_verifier_updates=0,ordinary_verifier_updates=0,frozen_MLP_updates=0,GMT_updates=0,
        reason='No Q2 training without certified two-action supervision',scientific_success=False)
    save(REPORTS/'TRAINING_METRICS.json',training)
    references=dict(GMT=ref(legacy.REPORTS/'ORIGINAL_GMT_METRICS.json'),frozen_MLP=ref(legacy.REPORTS/'ONLINE_mlp_learned.json'))
    save(REPORTS/'ONLINE_METRICS.json',dict(status='FROZEN_REFERENCES_REUSED_NEW_VERIFIERS_NOT_RUN',references=references,
        Q1_native_decisions_changed=0,typed_verifier=None,ordinary_verifier=None,
        reason='Q2 prerequisite failed' if complete and not eligible else 'Q2 prerequisite or implementation pending',scientific_success=False))
    save(REPORTS/'SPEED_METRICS.json',dict(status='NO_CURRENT_ROUND_COMPLETE_IMAGE_FPS_MEASUREMENT',
        historical_references={k:ref(legacy.REPORTS/f'SPEED_{k}.json') for k in ['original','mlp_learned','mlp_learned_cached']},
        audit_is_not_deployment_FPS=True,new_Q1_head_parameters=899,Q1_deployed=False,typed_parameters=None,
        MATCH_pre_audit_extra_association_solves_per_packet=4,
        native_closure_extra_association_solves='Two helper solves per packet, plus two real native association calls per disagreement; native Bank solves occur as needed within each replay.',
        scope='Offline audit instrumentation and state cloning costs are not deployed verifier costs.',scientific_success=False))
    save(REPORTS/'GO_NO_GO.json',dict(status=goal['status'],decision=decision,scientific_success=False,
        JEV_VERIFIER_INDEPENDENT_VALUE='NOT_ESTABLISHED',Q2_eligible=eligible,Q2_support=support,
        preliminary_accepted_MATCH_only_support=audit['Q2_support'] if audit else None,
        complete_native_plan_evidence=ref(REPORTS/'NATIVE_PAIRED_PLAN_OPPORTUNITY.json') if closure else None,
        Q1_diagnostic_only=q1,KEEP_GMT_MLP=True,evidence=ref(REPORTS/'PAIRED_PLAN_OPPORTUNITY.json') if complete else None,
        resource=ref(REPORTS/'RESOURCE_ACCOUNTING.json'),remaining_scientific_requirements=[
        'Certified two-action supervision and real typed native control','Fair trained ordinary verifier comparison',
        'Structural and cross-camera ablations','Measured identity/efficiency improvement beyond frozen MLP',
        'Qualified independent full-video and training-seed replication']))
    rows=['| 冻结历史参考 | HOTA | AssA | CVIDF1 | MOTA | IDSW | Frag |','|---|---:|---:|---:|---:|---:|---:|']
    for label,result in [('GMT',base),('GMT+MLP',dict(mlp['primary_raw']))]:
        m=result['metrics'];rows.append(f"| {label} | {m['HOTA']:.6f} | {m['AssA']:.6f} | {result['CVIDF1']:.6f} | {m['MOTA']:.6f} | {m['IDSW']:.0f} | {m['Frag']:.0f} |")
    q2lines=['| Q2动作（非时间留出） | 认证完整计划 | 人物连接依赖组 | video-person组 | 视频 |','|---|---:|---:|---:|---|']
    for action,r in support.items():q2lines.append(f"| {action} | {r['plans']} | {r['person_dependency_components']} | {r['unique_video_person_groups']} | {r['videos']} |")
    q1lines=[]
    if audit:
        q1lines=['| Q1类别 | 过去候选身份遭遇次数（有依赖） | 不同video/nativeID组 |','|---|---:|---:|']
        for label in ['RELIABLE','CONFLICTED','UNKNOWN']:
            q1lines.append(f"| {label} | {counts.get('Q1_'+label,0)} | {audit['Q1_unique_native_ID_groups'].get(label,0)} |")
    nc=closure['counts'] if closure else {}
    unknown=nc.get('plan_UNKNOWN_NATIVE_PLAN',0)/max(1,nc.get('plans',0)) if closure else counts.get('plan_UNKNOWN_PLAN',0)/max(1,counts.get('plans',0))
    changed=nc.get('plan_UNKNOWN_NATIVE_PLAN',0)/max(1,nc.get('actual_paired_native_replays',0)) if closure else counts.get('plan_UNKNOWN_PLAN',0)/max(1,counts.get('plans',0)-counts.get('plan_EQUIVALENT_NO_CHOICE',0))
    eligibility=['# 同状态决策监督审计','',f'阶段：{decision}。完整TRAIN12/13/14全部审计：{complete}。',*q2lines,'',*q1lines,'',
        f'UNKNOWN占全部计划 {unknown:.4%}，占实际有分歧计划 {changed:.4%}。各行双方对错、合法候选覆盖/缺失与别名计数见 PAIRED_PLAN_OPPORTUNITY.json。',
        '认证单位为完整计划。MATCH层先把改变的拒绝行保留UNKNOWN；新增原生闭包实际重放Bank/出生/WRITE，只有两条真实执行结果及严格过去唯一目标均可认证才解除UNKNOWN。两层记录分别保留。人物连接组包含当前GT人物与两方案候选owner，不把持续多帧事件当独立正例。',
        '原始时间留出桶不用于Q2训练计数；过去GT认证跨桶连续，不是严格标签隔离。样本与分组元数据合计≤32MiB，不保存新完整训练库。',
        '已有Rule/ORACLE事件不能改成Frozen-MLP的APPLY标签；历史重审计见 HISTORICAL_EVENT_REAUDIT.json。',
        f"真实Q2标签梯度诊断：{closure.get('Q2_real_label_gradient_smoke',{}) if closure else '原生闭包待完成'}。这不等于训练成功；即使单一类别有梯度，也不能冒充双类独立监督充足。",
        '完整原生计划见 NATIVE_PAIRED_PLAN_OPPORTUNITY.json；原MATCH预审计及逐行四象限见 PAIRED_PLAN_OPPORTUNITY.json。']
    if complete:
        save(REPORTS/'DECISION_LABEL_ELIGIBILITY.json',dict(status='ELIGIBLE_FOR_BOUNDED_SMOKE' if eligible else 'INSUFFICIENT_INDEPENDENT_Q2_SUPERVISION',
            support=support,preliminary_accepted_MATCH_only_support=audit['Q2_support'],source=ref(REPORTS/'NATIVE_PAIRED_PLAN_OPPORTUNITY.json')))
        save(REPORTS/'SUPERVISION_ELIGIBILITY.json',dict(Q2_eligible=eligible,Q2_support=support,Q1_support=audit['Q1_unique_native_ID_groups'],
            native_plan_evidence=ref(REPORTS/'NATIVE_PAIRED_PLAN_OPPORTUNITY.json'),preliminary_evidence=ref(REPORTS/'PAIRED_PLAN_OPPORTUNITY.json'),scientific_success=False))
    write('DECISION_SUPERVISION_AUDIT.md',eligibility)
    architecture=['# 架构与条件准入','',
        '完整GMT和5119参数冻结MLP保留；P0仅在相同原始轨迹的pre-commit分数/候选/历史包上离线比较，不在推理中使用GT。',
        'Q1问题是过去身份历史是否可靠；Q2问题是保留完整GMT计划还是完整MLP计划。完整双问题架构只有监督合格后才允许训练、接入native提交。',
        f'当前Q2准入：{eligible}。899参数HistoryReliabilityHead仅实现24→32→3诊断；输入只保留因果历史通道，不实现Q2、不改变native commit。',
        '不得把独立Q1分类头称为完整Typed JEV。Q1→Q2作用、问题共享表示和OFF全视频部署合同尚无新证据。原生双分支合同证明合法分数计划可实际改变native commit，不能冒充已经训练的Typed选择器。']
    write('ARCHITECTURE.md',architecture)
    q1metrics=[]
    if q1 and q1['status']=='COMPLETE_Q1_DIAGNOSTIC_ONLY':
        matrix=q1['confusion_true_by_pred'];per_class={}
        q1metrics=['','| Q1视频14诊断类 | Precision | Recall | F1 | 真值样本 |','|---|---:|---:|---:|---:|']
        for c,label in enumerate(q1['classes']):
            truth=sum(matrix[c]);predicted=sum(row[c] for row in matrix);tp=matrix[c][c]
            p=tp/predicted if predicted else 0.;r=tp/truth if truth else 0.;f=2*tp/(truth+predicted) if truth+predicted else 0.
            per_class[label]=dict(precision=p,recall=r,F1=f,truth_rows=truth,predicted_rows=predicted)
            q1metrics.append(f'| {label} | {p:.4%} | {r:.4%} | {f:.4%} | {truth} |')
        save(REPORTS/'Q1_CLASSIFICATION_DIAGNOSTICS.json',dict(per_class=per_class,
            accuracy=sum(matrix[i][i] for i in range(3))/sum(map(sum,matrix)),
            macro_F1=sum(v['F1'] for v in per_class.values())/3,source=ref(REPORTS/'Q1_RELIABILITY_DIAGNOSTIC.json'),
            independent_validation=False,native_decisions_changed=0,confidence_calibrated=False,scientific_success=False))
        q1metrics+=['',f"预设softmax≥0.9的CONFLICTED预测数：{q1['high_confidence_conflicted_predictions']}；固定外观代理预测数：{q1['appearance_consistency_proxy']['predictions']}。没有按结果调低阈值。",
            'Precision/Recall按采样诊断行计算，不是独立人物统计显著性或已校准置信度。历史分类不等于在线保护；该头从未参与native commit。']
    write('TRAINING_RESULTS.md',['# 真实训练记录','',f"Q1阶段：{training['status']}。Typed/ordinary/Q2/GMT/原MLP新增更新均为0。",'',
        'Q1若训练只使用TRAIN12/13非留出桶，固定300步；TRAIN14只作Q1开发诊断。当前GT不作为历史特征，UNKNOWN仅为过去证书不充分的三分类标签，不是动作伪标签。','',
        '完整逐25步loss/梯度、权重SHA、类别支持与混淆矩阵见 TRAINING_METRICS.json / Q1_RELIABILITY_DIAGNOSTIC.json。',
        'Q1分类成绩不表示已改善HOTA、通过Q2机制或实现因果身份保护。',*q1metrics])
    write('ONLINE_EVALUATION.md',['# 在线结果与冻结参考','',*rows,'',
        '本表均为冻结V1完整TRAIN16原生raw结果，已暴露开发集。新Typed/ordinary verifier在线评测尚未运行；没有新HOTA差值。',
        '原GMT与MLP第一次分歧ID52/25仍为UNKNOWN；本轮不把它重新认定为纠错，也未执行新的完整视频配对干预。',
        f'P0原GMT完整TRAIN预测复核通过段数：{len(audit["cases"]) if audit else 0}。该证据不冒充新Verifier OFF全视频部署合同。'])
    write('EFFICIENCY.md',['# 效率证据边界','',
        '冻结历史同GPU7完整图像FPS：GMT3.409768；MLP3.129765；缓存MLP2.848982。默认和缓存均失败95%门槛。这是V1测量，本轮未产生新的完整FPS。',
        'P0每个包有额外诊断求解与GT标注，不用其帧率冒充部署速度。当前只训练/诊断Q1899参数，它未接入部署；新Typed开销、参数与FPS均NOT_RUN。',
        '原MLP24帧GPU组件诊断显示feature p50约9.16ms、head约.537ms，仅为前缀定位；不外推全视频，不因此声称缓存更快。',
        '只有新模型真实完整在线质量合格后才启动新同卡完整图像效率测量；旧九次交错方案明确未执行。',
        '资源账本：'+json.dumps(resources,ensure_ascii=False)])
    write('QUESTION_ABLATION.md',['# 问题结构消融','',
        '未运行：去Q1、去Q2、去类型结构、去跨相机历史。','',
        '若Q2监督准入失败，以上按预注册条件不启动，不能用Q1分类成绩替代必要性消融。当前与普通验证器的结构独立比较NOT_ESTABLISHED。'])
    reviewer=['# 审稿人证据与风险','',
        'R1：过去身份混杂是潜在需求；当前GMT/MLP计划中可认证纠错空间以P0支持量为准，尚未证明需要新验证器。',
        'R2：没有训练公平普通验证器就不能宣称Typed结构胜出；P0失败时不浪费预算拟合伪标签。',
        'R3：Q1诊断同时列固定外观一致性代理规则；不是GT纯度部署阈值，也没有在线收益证明。',
        'R4：目标是共享历史的可靠性问题与合法计划选择问题；当前只有Q1诊断时，不声称已实现双问题真实决策。',
        'R5：缺少Q1/Q2必要性、类型结构与跨相机消融，结构贡献尚未建立。',
        'R6：跨相机owner冲突数量独立报告；有跨相机输入不等于MCMT创新已被证明。',
        'R7：旧MLP HOTA/AssA/CVIDF1改善，但IDSW增加47，Frag减少2、MOTA下降；历史持续量不是单次干预的因果传播时间。',
        'R8：原MLP特征维护与同步开销高于小头；缓存未观察到加速。没有新同环境完整FPS，不能仅凭899参数称实时。',
        'R9：GOT-JEPA将JEPA改造成跟踪模型预测，clean/corrupted当前帧的师生共享历史并学习模型适应；本项目拟做的是冻结GMT上的历史可靠性和合法计划选择，没有实现其预训练或OccuSolver，也不借用其成绩。方法论类比仅是把思想落到具体能力。[原论文v5](https://arxiv.org/abs/2602.14771v5)。',
        'R10：尚无跨视频身份检索/实体链接/视频索引任务及实验。MCMT场景可关联这些Web应用，但当前没有相应证据，不能添加Web-scale字样充当WWW贡献；投稿定位风险仍高。']
    write('REVIEWER_RISK_ANALYSIS.md',reviewer)
    report=['# GMT-JEV V2 中文研究报告','',f'当前科学判断：**{decision}**；SCIENTIFIC_SUCCESS=false。','',
        'A. 分支 research/gmt-jev-typed-verifier-20261015；当前Goal状态 '+goal['status']+'。准确发布HEAD以runtime/publication/LAST_VERIFIED.json为准；本报告生成HEAD为 '+subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()+'，不冒充未来发布提交。',
        'B. 完整GMT与原MLP step300文件/源码SHA冻结。P0原GMT完整TRAIN记录逐项复核见各case；原MLP旧严格证据复用，未声称运行了新Typed OFF全视频合同。','',
        'C. Q1使用既有已提交历史，输出RELIABLE/CONFLICTED/UNKNOWN；Q2拟在同状态下选择完整KEEP/APPLY计划。若P0失败，本轮只做Q1诊断，没有Q2训练或新native决策。','',
        'D. 真实监督支持：',*q2lines,'',*q1lines,'',
        f'UNKNOWN计划比例：全部 {unknown:.4%}；实际分歧中 {changed:.4%}。完整逐行四象限、候选缺失、别名和Q2梯度诊断见机器记录。','',
        'Q1诊断（不参与在线身份提交）：',*q1metrics,'',
        'E. 同条件冻结历史成绩：',*rows,'',
        'MLP相对GMT：HOTA+0.600573、AssA+1.034080、CVIDF1+1.456462、IDSW+47、Frag−2，MOTA约−0.0718。这些均是旧开发结果，不归给新Q1/Typed。','',
        'F. 相对现有MLP，本轮尚未建立新的在线收益。Q1诊断不改变任何提交；无法给出新HOTA/IDSW改善值。','',
        'G. Typed相对普通验证器的独立贡献NOT_ESTABLISHED；Q2未准入时二者均不训练，没有用弱对照包装胜出。','',
        'H. 历史完整图像FPS为GMT3.409768、MLP3.129765、缓存MLP2.848982。新Verifier FPS尚未测；不将P0审计或CPU小头速度冒充完整FPS。','',
        'I. TRAIN16已暴露；Q1 TRAIN14诊断也不属于合格独立视频。独立完整视频与必要训练种子仍PENDING。','',
        'J. 现有证据尚不足以支撑双问题Typed-JEV的WWW创新主张；最缺可靠的两动作监督、真实结构/机制增量和独立任务验证。详细R1–R10见 REVIEWER_RISK_ANALYSIS.md。','',
        '本轮资源：'+json.dumps(resources,ensure_ascii=False),'',
        '失败处理：若全部授权TRAIN与既有日志仍无足够KEEP/APPLY监督，按任务书停止本轮结构探索，保留GMT+MLP。不会因还有GPU预算而增加Question、制造标签或改用封存数据。','',
        f'最终路线：**{decision}**。原MLP继续作为最强已测开发参考；科学目标未完成。']
    write('FINAL_RESEARCH_REPORT_ZH.md',report)
    print('V2_REPORT',decision,'Q2',support,'GPU_hours',spent(),flush=True)
if __name__=='__main__':main()

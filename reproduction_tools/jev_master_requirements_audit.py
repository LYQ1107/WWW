"""Scope-aware evidence audit; present artifacts never prove master scientific success."""
from jev_master_audit_common import *
import re,collections
SOURCE=Path('/home/liuyeqiang/WWW_jev_phase21')
REQUEST=Path('/home/liuyeqiang/.codex/attachments/59faae16-96ec-4d79-9660-e899dc257d8f/Pasted text.txt')
def main():
 protect();result=read(MASTER/'COMMON_CACHE_GTA_RESULTS.json');assert len(result['cases'])==12
 raw=read(OUT/'RESULT.json');assert raw==result
 actor=result['binding']['actor_commit'];code=result['binding']['actor_source']
 assert hashlib.sha256(subprocess.check_output(['git','show',actor+':reproduction_tools/jev_master_common_cache_gta.py'],cwd=ROOT)).hexdigest()==code['SHA256']
 assert sha(code['path'])==code['SHA256']
 for f in result['source_files']+[result['model_config']]:assert sha(f['path'])==f['SHA256']
 verified={}
 for phase in [17,18,19,20,21]:
  root=Path('/home/liuyeqiang/WWW_jev_phase'+str(phase));m=read(root/f'reports/JEV_PHASE{phase}/ARTIFACT_SHA_MANIFEST.json')
  for n,h in m['files'].items():assert sha(root/n)==h,(phase,n)
  verified[str(phase)]=dict(manifest=ref(root/f'reports/JEV_PHASE{phase}/ARTIFACT_SHA_MANIFEST.json'),publication_files=len(m['files']))
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==BASE
 gate=read(SOURCE/'reports/JEV_PHASE19/LEARNING_ELIGIBILITY.json');assert not gate['qualified'] and (gate['safe_TRAIN'],gate['safe_RESERVED'])==(3,1)
 prior=read(MASTER/'COMMON_CACHE_GTA_ATTEMPT2_PARTIAL.json')
 assert all(a['raw_GTA_scores']==b['raw_GTA_scores'] and a['predictions']==b['predictions'] for a,b in zip(prior,result['cases']))
 equals=sum(c['predictions']['common_cache_GTA']['ref']==c['predictions']['four_slot_cosine']['ref'] for c in result['cases'])
 assert equals==11
 save(MASTER/'COMMON_CACHE_GTA_INTERPRETATION.json',dict(status='BOUNDED_POINTWISE_DIAGNOSTIC_COMPLETE',
  actor_commit=actor,results=ref(MASTER/'COMMON_CACHE_GTA_RESULTS.json'),GTA_and_cosine_same_top1=equals,total=12,
  available_correct_owner_rows=sum(c['rank_of_correct_owner'] is not None for c in result['cases']),
  GTA_over_cosine_new_correct_rows=0,summary=result['summary'],
  comparisons_NOT_equivalent='GTA/cosine are point-wise ranks; frozen XV is an actual joint assignment. No same-solver superiority inference.',
  sampling='Original12previouslyknown target rows in3TRAIN videos, not12certified independent experiments.',
  original_GMT_control='NOT_RECONSTRUCTED',new_training=False,new_native_actions=0,
  safety_gate_unchanged=ref(SOURCE/'reports/JEV_PHASE19/LEARNING_ELIGIBILITY.json'),independent_value='NOT_ESTABLISHED'))
 src=SOURCE/'gtr/modeling/roi_heads/gtr_roi_heads.py';cache=SOURCE/'reproduction_tools/cache_jev_phase13_stage1.py'
 feasibility=dict(status='ORIGINAL_D0_NOT_RECOVERABLE_FROM_SAVED_VFCE_ONLY_INPUTS',
  sources=[ref(src),ref(cache),ref(SOURCE/'docs/JEV_PHASE13_GMT_AUDIT.md')],
  actual_prefix_dimensions=[dict(event=c['event_key'],VFCE=c['available_cache_dim'],adapted_GTA=c['GTA_input_dim'],excluded_camera=c['uncommitted_same_scene_camera_excluded']) for c in result['cases']],
  original_ROI_path='jitter_bboxes_center -> ROI pooling -> VFCE + RPCE128',
  saved_cache_path='unjittered proposal ROI pooling -> VFCE1024; GTA/RPCE execution explicitly forbidden',
  mismatch='Original jittered ROI/RPCE tensors not stored. Rebuilding geometry on saved unjittered boxes is a distinct common-cache adapter.',
  scorer='Verified trained original Stage2 model20000 association module, rather than assuming Stage1 REID weights are a qualified GTA scorer.',
  exact_original_D0='NOT_RUN',adapted_common_cache_control='COMPLETE_POINTWISE_ONLY',
  future_original_replay_requires='Explicit source-aligned perception/extraction protocol and trained association checkpoint; report any perception change separately. No retrospective relabel of the adapter.')
 save(MASTER/'ORIGINAL_GMT_INPUT_FEASIBILITY.json',feasibility)
 # Every original nonempty clause remains traceable; uncertain clauses never pass by artifact presence.
 lines=REQUEST.read_text().splitlines();sections={};section=0;clauses=[];paths={}
 for i,line in enumerate(lines):
  t=line.strip();m=re.match(r'^(\d{1,2})\. (.+)$',t)
  if m and i>0 and lines[i-1].startswith('===='):
   section=int(m.group(1));sections[section]=m.group(2);continue
  if not t or t.startswith(('====','#')):continue
  path=None
  if re.fullmatch(r'(?:docs|reports|reproduction_tools)/[^\s]+\.(?:md|json|py)',t):path=t
  elif re.fullmatch(r'PHASE17_[A-Z_]+\.(?:json|md)',t):path=('reports/JEV_PHASE17/' if t.endswith('.json') else 'docs/')+t
  if path:
   p=SOURCE/path;item=dict(path=path,exists=p.exists(),source_lines=[],content_scope='Artifact content must be assessed separately; existence is not scientific completion')
   if p.exists():
    item['file']=ref(p)
    if p.suffix=='.json':item['declared_status']=read(p).get('status','NO_TOP_LEVEL_STATUS')
   paths.setdefault(path,item)['source_lines'].append(i+1)
  clauses.append(dict(source_line=i+1,section=section,requirement=t,evidence_group='S'+str(section),
   assessment='ARTIFACT_PRESENCE_ONLY' if path else 'SEE_SCOPE_AND_UNRESOLVED_REQUIREMENTS;NOT_MASTER_COMPLETION_PROOF'))
 required_fields=['goal_id','final_scientific_objective','target_conference','frozen_reference','scientific_success_criteria','web_relevance','permitted_data_scope','forbidden_data_scope','current_phase','current_phase_status','master_status','active_hypotheses','rejected_hypotheses','evidence_registry','artifact_registry','next_experiment','remaining_authorized_budget','conditions_for_final_evaluation','paper_readiness']
 fields=read(MASTER/'MASTER_GOAL.json');assert all(n in fields for n in required_fields)
 groups=[]
 def add(s,scope,status,evidence,remaining):
  refs=[ref(SOURCE/p) for p in evidence];groups.append(dict(id='S'+str(s),section=sections.get(s),required_scope=scope,assessment=status,evidence=refs,unresolved=remaining))
 add(0,'Unified reliable MATCH/REACT/WRITE/ownership/shared computation, independent gains and paper', 'NOT_ACHIEVED', ['reports/JEV_PHASE21/FINAL_GO_NO_GO.json','reports/JEV_PHASE19/LEARNING_ELIGIBILITY.json'], 'Schema is present; model capability, independent value and submission readiness are not proven.')
 add(1,'Preserve historical weights/states/predictions/negative findings and immutable branches', 'PUBLICATION_INTEGRITY_VERIFIED;WEIGHT_PRESERVATION_SCOPE_LIMITED', ['reports/JEV_PHASE17/FROZEN_PRIOR_INTEGRITY.json'], '535-reference old audit exists; this run rehashes publication files and one trained scorer, not all historical339weights/rawstates. No broad current all-file completion claim.')
 add(2,'Six source-level repository/function/license/IO/training audits', 'SOURCE_AUDIT_AVAILABLE;RUNTIME_AND_NATIVE_PORTABILITY_NOT_PROVEN', ['reports/JEV_PHASE17/EXTERNAL_CODE_MANIFEST.json','docs/PHASE17_VISUAL_JEV_SOURCE_AUDIT.md','docs/PHASE17_EXTERNAL_ARCHITECTURE_MATRIX.md'], 'Source reading does not prove six deployed native engines or dynamic>16 identity choice.')
 add(3,'Frozen WHO/Q4 full-video2x2, matched-prefix causal feedback and first divergence', 'VERIFIED_FROZEN_DIAGNOSTIC;NOT_NEW_POLICY_OR_INDEPENDENT_VALIDATION', ['reports/JEV_PHASE17/WHO_Q4_FACTORIAL.json','reports/JEV_PHASE17/FIRST_DIVERGENCE.json','reports/JEV_PHASE17/PHASE17_COLD_START_VS_PREFIX.json','reports/JEV_PHASE17/PREFIX_WHO_Q4_FACTORIAL.json'], 'B0/B1/B2/B3 IDSW147/267/330/1494 and96prefix branches explain frozen mechanism; final independent system remains untested.')
 add(4,'Observable owner evidence, allC1-C6, independent labels and valid candidate-wise action value', 'BOUNDED_DIAGNOSTIC_AVAILABLE;TRAINING_ELIGIBILITY_FAILED', ['reports/JEV_PHASE17/OWNER_IDENTIFIABILITY.json','reports/JEV_PHASE19/LEARNING_ELIGIBILITY.json'], 'C4 numerical ambiguity proxy does not prove information impossibility. Native66branches yield3TRAIN/1RESERVED, below8/3. Unexecuted utility stays null.')
 add(5,'D0-D4 teacher controls, full/cache/batch/order, true dynamic candidates, calibration and cache invalidation', 'PARTIAL;FROZEN_POPULATION_FP32_CACHE_PASS_ONLY', ['reports/JEV_PHASE17/NO_NEW_HEAD_ABLATION.json','reports/JEV_PHASE17/VISUAL_JEV_COMPARISON.json','reports/JEV_PHASE21/VISUAL_PROFILE_ATTRIBUTION.json'], 'OriginalD0 absent, new adapter distinct. No trained head/taskLoRA/same-imageMLP controls. Order18/24 only; repeated-option stress not dynamicidentity ranking; no native persistent serving cache.')
 add(6,'Qualified causal soft/owner/visual weighting and complete cold-start safety', 'TWO_SOFT_RULES_REJECTED;OWNER/VISUAL_STRATEGIES_UNQUALIFIED', ['reports/JEV_PHASE17/SOFT_MEMORY_CAUSAL_RESULTS.json','reports/JEV_PHASE18/FINAL_GO_NO_GO.json'], '48softbranches harmful; full newstrategy cold runs NOT_RUN by safety gate. E3/E4 lack reliable independent evidence; not an established good memory controller.')
 add(7,'Qualified tiny/pilot/formal F1/F2 learning, lawful native lifecycle and executed causal value', 'NOT_RUN_LEARNING_GATE', ['reports/JEV_PHASE17/TRAINING_RESULTS.json','reports/JEV_PHASE19/LEARNING_ELIGIBILITY.json'], 'No qualifying newcontroller or identityLoRA training; correctness, joint utility and censored UNKNOWN must remain separate.')
 add(8,'Same evidence/labels/solver/budget G0-G11 and shared versus separate models', 'NOT_RUN_FAIR_LEARNED_COMPARISON', ['reports/JEV_PHASE17/FAIR_BASELINE_RESULTS.json'], 'OriginalGMT reference is a different full frontend; pointwise GTA/cosine versus nativejoint JEV is not fair architecture ablation. MLP/DeepSets/SetTransformer and lifecycle controls still missing.')
 add(9,'Qualified newpolicy full native validation, pooled tracking/crosscamera metrics, harm and endtoend25FPS', 'FROZEN_DIAGNOSTICS_AVAILABLE;NEW_POLICY_AND_DEPLOYMENT_NOT_PROVEN', ['reports/JEV_PHASE17/ONLINE_TRACKING_RESULTS.json','reports/JEV_PHASE17/LATENCY_RESULTS.json','reports/JEV_PHASE21/VISUAL_EXECUTION_LATENCY.json'], 'ReusedDEV is not independent validation. Native256scene oldprofile~3.25FPS; teacherN32microbench is not onlineMOT/p95. No newpolicy full-video causal safety/latency qualification.')
 add(10,'Real reliable identity-index retrieval, wrong segments/duration/latency/update and Webpaper task', 'PARTIAL_CAUSAL_INDEX_DIAGNOSTIC;NO_DEPLOYED_LEARNED_POLICY', ['reports/JEV_PHASE20/NATIVE_ACTION_RETRIEVAL_VALUE.json','reports/JEV_PHASE20/NATIVE_RETRIEVAL_TEMPORAL_DECOMPOSITION.json','reports/JEV_PHASE17/IDENTITY_INDEX_RETRIEVAL.json'], 'Three safe oracleactions in two dependent units reduce laterwrongreturns without laterID changes. Not learned policy, independent complete video, production index or latency qualification.')
 add(11,'Frozen science/data/safety/runtime/version gates with separate status axes', 'CURRENT_FAILED_GATES_PRESERVED', ['reports/JEV_PHASE19/FINAL_GO_NO_GO.json','reports/JEV_PHASE21/FINAL_GO_NO_GO.json'], 'ScientificGO/deployment/independentvalue absent; failed old gates cannot be changed after outcomes.')
 add(12,'Recoverable evidence-driven execution, bounded hypotheses, failures kept and next qualified task', 'CONTROLLER_AND_BOUNDED_EXECUTION_PRESENT;MASTER_ACTIVE_PIVOT', ['reproduction_tools/jev_master_goal_controller.py','reports/JEV_MASTER/NEXT_ACTION.json'], 'No eligible learning task currently ready; metadata source question pending. Do not restart completedGPU work or invent endless prompt/policy versions.')
 add(13,'Authorized splits/resources/environments/GTfuture protection and compact uploads', 'CURRENT_DIAGNOSTIC_WITHIN_EXISTING_SCOPE;NEW_SOURCE_PENDING', ['docs/PHASE20_NEXT_DATA_SCOPE_PROPOSAL.md','reports/JEV_PHASE19/FINAL_GO_NO_GO.json','reports/JEV_PHASE21/GPU_EXECUTION.json'], 'Approved2-9/reserved10/11/15 extension is completed. Newsource A only proposed, no answer/path yet. OfficialTEST/Full24/sealed20-22 stay closed; no newimages/training approved by the pending proposal.')
 add(14,'Verifiable submission readiness, truthful independent science and complete paper', 'NOT_READY', ['docs/WWW_2027_SUBMISSION_READINESS.md','docs/WWW_JEV_PAPER_READINESS.md'], 'No final paper/package, independent learned gains, fair tables or qualified deployment. Earlier deadline citation is not newly verified by this audit.')
 add(15,'P0-P11 progression with dependent training blocked but independent diagnostics performed', 'PARTIAL_PROGRESS_WITH_DATA_BARRIER', ['reports/JEV_PHASE19/FINAL_GO_NO_GO.json','reports/JEV_PHASE20/FINAL_GO_NO_GO.json','reports/JEV_PHASE21/FINAL_GO_NO_GO.json'], 'Source/factorial/owner/teacher/soft diagnostics executed; newlearning/fair controls/independent unified evaluation absent. Phase completion cannot close master.')
 add(16,'All named deliverables, honestNOT_RUN, Q1-Q12, tests/provenance/push', 'NAMED_DELIVERABLES_INSPECTED;SCIENTIFIC_COMPLETION_NOT_PROVEN', ['docs/PHASE17_FINAL_RESEARCH_REPORT.md','reports/JEV_PHASE17/VERIFICATION_RESULTS.json','reports/JEV_PHASE17/FINAL_GO_NO_GO.json'], 'Artifact existence and successful checks cover bounded implementations; Q8-Q11 scientific gains/unified necessity remain unproven, Q6native benefit not established.')
 add(17,'Construct unified reliable online system with independent contribution, without gate relaxation', 'NOT_ACHIEVED', ['reports/JEV_PHASE21/FINAL_GO_NO_GO.json'], 'Do not mark goal complete. Main next dependency is manually valid independent supervised/validation population and subsequent actual native safety qualification.')
 # Frozen source files are checked against their snapshots, not inferred from README.
 external=read(SOURCE/'reports/JEV_PHASE17/EXTERNAL_CODE_MANIFEST.json');sources=[]
 for repo in external['repositories']:
  checked=0
  for file in repo['files']:
   p=Path(repo['source_path'])/file['path'];assert sha(p)==file['SHA256'];checked+=1
  sources.append(dict(name=repo['name'],commit=repo['commit'],license=repo['license'],source_files_SHA_verified=checked))
 audit=dict(status='MASTER_COMPLETION_NOT_PROVEN',source_request=ref(REQUEST),source_root=str(SOURCE),source_commit=BASE,
  all_source_sections=sections,clauses=clauses,named_deliverables=list(paths.values()),master_required_fields={n:n in fields for n in required_fields},
  evidence_groups=groups,frozen_publications_rehashed=verified,external_sources_rehashed=sources,
  scope_rule='Every nonempty original specification clause is indexed. Section evidence is scoped; a complete artifact/manifest is not a proof of broad scientific completion.',
  final_goal_complete=False,scientific_success=False,learning_gate=ref(SOURCE/'reports/JEV_PHASE19/LEARNING_ELIGIBILITY.json'),
  new_control=ref(MASTER/'COMMON_CACHE_GTA_INTERPRETATION.json'),original_GMT_reconstruction=ref(MASTER/'ORIGINAL_GMT_INPUT_FEASIBILITY.json'),
  next_qualified_task='Audit official public release source-code and split manifests now; no dataset labels/images. Dataset-label scope and private MMPTRACK path remain pending, not a barrier to public source-document review.',
  all_explicit_requirement_rows_reviewed_for_scope=True,unknown_or_indirect_evidence='NOT_ACHIEVED',new_training=False,new_native_actions=0)
 save(MASTER/'MASTER_COMPLETION_REQUIREMENTS_AUDIT.json',audit)
 text='''# MASTER GOAL 完成性与共同缓存 GTA 对照核对

长期科研目标尚未完成。本轮逐项回看原始MASTER规范，所有非空条款均在JSON索引中保留原行号、原文与对应证据组；不以文件存在、检查通过或阶段数量代替完整科学证据。旧V–XXI分支和结果保持冻结。

## 新增的可执行数值对照

原D0“OriginalGMT/GTA”仍不能从现有缓存原样恢复：原eval路径对ROI框加入jitter，然后提取VFCE并拼接RPCE128；当前缓存是独立冻结的无jitter VFCE1024，明确禁用GTA/RPCE。Stage1 REID与已训练Stage2关联模块也不能混为同一权重资格。源码、配置和实际输入维数见ORIGINAL_GMT_INPUT_FEASIBILITY.json。

因此单独预注册共同缓存GTA诊断：全部原12teacher事件，不按新分数重选；保留原VFCE，使用未抖动的真实框、邻域几何和真实camera/2输入已训练s_t_head，再调用原get_asso、原per-frame unmatched softmax和按历史预测ID聚合的GTA分数。所有候选均来自原合法窗口。每次恢复原trajectory RNG；两次评分逐项完全相同。没有ROI重提取、训练或真实新身份提交。

| 方法 | 选择机制 | 正确owner | 错误owner | UNKNOWN |
|---|---|---:|---:|---:|
| 共同缓存GTA | 当前行点式Top1 | 5 | 2 | 5 |
| 四槽余弦 | 当前行点式Top1 | 5 | 1 | 6 |
| 冻结XV JEV | 原实际联合assignment | 6 | 2 | 4 |

GTA与余弦11/12次选同一ID；另一行是余弦UNKNOWN→GTA确认错误，没有新增正确owner。7/12行具备原XVII规则认证的合法正确owner，另5行没有；不能把UNKNOWN写成错误或安全纠错。旧XVII owner定义与XVIII以后的immutable-owner训练认证不同，本轮只复核旧人口，绝不升级3TRAIN＋1预留Gate。

video16帧21行0，GTA与余弦点式正确而原JEV联合选择错误；video16帧19行28则相反，点式选错而原联合选择正确。这正说明点式排名不能替代联合容量分配或未来动作价值。5与6不能当作同求解器架构比较，原始完整GMT也不能用本adapter替代。12行仅覆盖3个此前反复使用的TRAIN视频，且有同帧/同人相关性，不宣称12个独立试验或泛化显著性。视觉teacher蒙太奇与数值历史证据还具有不同表示与成本，正式同证据公平实验仍NOT_RUN。

首轮将raw VFCE误与normalized JEV输入直接比较，未产生分数就失败。修复1保留raw GTA输入，并验证其归一化后与原JEV输入对应；同时适配旧query的key字段。第二轮因误以为camera0前缀只包含一路当前检测而断言失败，前2组分数完整保留。修复2只更正总长度断言，原因果切片未变：camera0评分明确排除同场景未提交camera1。最后12组全部完成，前2组分数与旧部分输出精确一致。两次普通工程修复、失败及对应Git提交全部保留，没有修改科学设置或选择有利版本。

成功执行约21.7秒，单张空闲GPU2，峰值allocated显存约366.2MiB；评分本身延迟与全部原始数值在RESULTS。仅复用既有权重，没有新模型、数据图像或环境。大文件留服务器，GitHub仅提交必要代码、协议、JSON及此报告。

## 原任务要求与当前证据

| 原规范部分 | 当前证据的范围 | 尚未完成的要求 |
|---|---|---|
'''
 for g in groups:text+='| '+str(g['id'])+' '+str(g['section'])+' | '+g['assessment']+' | '+g['unresolved'].replace('|','/')+' |\n'
 text+='''
完整条款/命名文件/证据SHA见MASTER_COMPLETION_REQUIREMENTS_AUDIT.json。重新核验XVII–XXI发布清单与六仓库所读源码，只证明冻结文件/快照完整，不自动证明模型能力；本轮没有重hash所有历史339权重或遍历全部原始native状态。当前对最终模型不确定、间接、缺失的证据均按未完成处理。

## 下一步与授权边界

现在主要缺口是独立人工有效的身份及长期动作监督、可信验证关系，并在通过资格后进行相同证据/求解器/监督预算的强基线和完整原生评价。共同缓存GTA没有改变这个依赖，也不是新的成功模型。当前论文NOT_READY、独立JEV价值NOT_ESTABLISHED；Goal继续ACTIVE/PIVOT_REQUIRED。

新来源A阶段方案已在PhaseXX输出，问题仍待范围/合法MMPTRACK路径回复。这不阻止继续审计官方公开的发布代码和分割清单；该只读来源工作不读取新数据集标签/图像，不依赖训练资格。本轮只使用原获准的TRAIN12/13/16事件；未读新增来源、sealed20/21/22或官方TEST，不启动Full24。原MASTER第13节要求：“如果需要额外训练视频或新的独立数据集，先输出研究必要性、预注册划分与资源影响”；此前追加批准仅覆盖VisionTrack02–09及预留10/11/15，本轮没有把它扩成新数据集或图像/训练授权。

复现：先核对COMMON_CACHE_GTA_PROTOCOL、既有checkpoint和prefix SHA；在独立GMT环境以CUDA_VISIBLE_DEVICES=空闲GPU运行reproduction_tools/jev_master_common_cache_gta.py，选择新的JEV_MASTER_GTA_OUT目录。禁止覆盖COMPLETE输出。运行jev_master_requirements_audit.py用于重新核验静态规范、固定源码和发布证据；控制器应将公开来源代码/分割清单审计列为下一项待实现任务；它不会启动不合格训练。
'''
 (ROOT/'docs/WWW_JEV_MASTER_REQUIREMENTS_AUDIT.md').write_text(text)
 goal=read(MASTER/'MASTER_GOAL.json');goal.update(current_phase_status='MASTER_REQUIREMENTS_REVIEWED_COMMON_CACHE_DIAGNOSTIC_COMPLETE',master_status='PIVOT_REQUIRED',
  next_experiment='PUBLIC_RELEASE_METADATA_AUDIT',active_hypotheses=[],scientific_SUCCESS=False,paper_readiness='NOT_READY')
 goal['remaining_authorized_budget']=dict(completed_diagnostic='COMMON_CACHE_GTA_12_TARGET_ROWS',new_training=0,new_native_actions=0,new_models=0,new_source='Official source-code/split-manifest review is available; dataset labels/images and private source access remain pending',scope_note='Existing diagnostic fixed population finished; no automatic new science versions or data expansion.')
 save(MASTER/'MASTER_GOAL.json',goal)
 reg=read(MASTER/'EXPERIMENT_REGISTRY.json');reg['experiments']=[e for e in reg['experiments'] if e['id']!='MASTER_COMMON_CACHE_GTA']
 reg['experiments'].append(dict(id='MASTER_COMMON_CACHE_GTA',status='COMPLETE',requires=[],requires_learning_gate=False,commands=[],
  description='Distinct adapted numerical point-ranking control, not originalGMT/nativecausal/architecture comparison',actual_actor_commit=actor,evidence=[ref(MASTER/'COMMON_CACHE_GTA_PROTOCOL.json'),ref(MASTER/'COMMON_CACHE_GTA_RESULTS.json')]))
 reg['experiments']=[e for e in reg['experiments'] if e['id']!='PUBLIC_RELEASE_METADATA_AUDIT']
 reg['experiments'].append(dict(id='PUBLIC_RELEASE_METADATA_AUDIT',status='PENDING',requires=['MASTER_COMMON_CACHE_GTA'],requires_learning_gate=False,commands=[],description='Inspect official public MMPTRACK/WILDTRACK/SCOUT release code and split manifests; no dataset GT/images, no private messages or signatures',evidence=[ref(SOURCE/'docs/PHASE20_NEXT_DATA_SCOPE_PROPOSAL.md')]))
 save(MASTER/'EXPERIMENT_REGISTRY.json',reg)
 ev=read(MASTER/'EVIDENCE_REGISTRY.json');ev['entries']=[x for x in ev['entries'] if 'WWW_jev_master_audit/reports/' not in x['path']]
 ev['entries'].extend([ref(MASTER/'COMMON_CACHE_GTA_RESULTS.json'),ref(MASTER/'MASTER_COMPLETION_REQUIREMENTS_AUDIT.json')]);save(MASTER/'EVIDENCE_REGISTRY.json',ev)
 save(MASTER/'NEXT_ACTION.json',dict(status='READY_PUBLIC_RELEASE_SOURCE_REVIEW',scientific_success=False,
  next='Inspect official public release code/split manifests; no datasetGT/images, newnativebranches or training. Existing source-A label scope/path question remains pending.',
  proposal=str(SOURCE/'docs/PHASE20_NEXT_DATA_SCOPE_PROPOSAL.md'),completion_audit=str(MASTER/'MASTER_COMPLETION_REQUIREMENTS_AUDIT.json'),
  blocker='Independent supervision3TRAIN/1RESERVED below8/3 and no established independent source access. GTA diagnostic is complete; public release-code/split review remains available without label access.'))
 note='\n\nMASTER completion audit: all original clauses/sections/deliverables indexed with scoped evidence, not completion. Distinct adapted common-cache GTA12targets completed(5correct/2wrong/5UNKNOWN; cosine5/1/6; nativejointXV6/2/4). No matchedsolver superiority, extra independent labels, native benefit or training qualification. OriginalD0 remains not reconstructed. Twoengineering failures preserved; source-A scope/path question remains pending.\n'
 p=MASTER/'RESEARCH_DECISIONS.md'
 if note not in p.read_text():p.write_text(p.read_text()+note)
 print(json.dumps(dict(status=audit['status'],clauses=len(clauses),sections=len(groups),named_artifacts=len(paths),publications=verified,common_cache_GTA=result['summary'],master_complete=False),indent=2))
if __name__=='__main__':main()

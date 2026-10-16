"""Publish measured frozen evidence and explicit conditional nonexecution states."""
import collections,csv,gzip,hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
from omega_corr_common import ROOT,RUNTIME,sha256,provenance


def main():
    source=provenance();assert not source['source_status'].strip()
    def load(p):return json.loads(p.read_text())
    def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    def csvout(p,rows,fields=None):
        fields=fields or list(dict.fromkeys(k for r in rows for k in r));f=p.open('w',newline='');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.close()
    ev=load(RUNTIME/'evaluation_v1.json');review=load(RUNTIME/'control_review_v1.json');cycle=load(RUNTIME/'cycle_audit_v1.json');geo=load(RUNTIME/'geometry_support_v1.json');native=load(RUNTIME/'native_context_full_v1/manifest.json');qs=load(RUNTIME/'native_candidates_v1.json')['queries'];truth=load(RUNTIME/'candidate_certifications_v1.json')['rows'];assert ev['gateB_status']=='NO_UNIQUE_CROSSVIEW_TOKEN_EVIDENCE'
    results=ROOT/'results';docs=ROOT/'docs';configs=ev['configurations'];selected=ev['selected_layer_fit_only'];selected_name=f'FULL_IMAGE_MULTIVIEW|omega|F2|{selected}|M0|primary|0';selected_id=next(i for i,c in enumerate(configs) if c['config_id']==selected_name)
    # The failed necessary gate is sufficient to stop training. All independent
    # registered mechanisms have nevertheless been executed and reviewed.
    states=dict(TOKEN_CORRESPONDENCE_EVIDENCE='REAL_TOKEN_INTERFACE_PASS; DISCRIMINATIVE_ADVANTAGE_GATE_NOT_PASSED',UNIQUE_OMEGA_EVIDENCE='NO_UNIQUE_OMEGA_CORRESPONDENCE_EVIDENCE',NATIVE_GMT_CORRECTION='NOT_EXECUTED_GATE_B_FAILED',FULL_TRACKING_EFFICIENCY='NOT_EXECUTED / NOT_MEASURED')
    shutil.copyfile(RUNTIME/'derived_candidate_scores_v1.csv',results/'omega_corr_candidate_scores.csv')
    shutil.copyfile(RUNTIME/'paired_query_results_v1.csv.gz',results/'omega_corr_query_results.csv.gz')
    for src,dest in [('candidate_values_v1.npz','omega_corr_candidate_values.npz'),('candidate_values_v1.json','omega_corr_candidate_values_index.json')]:shutil.copyfile(RUNTIME/src,results/dest)
    for p in ['evaluation_v1.json','control_review_v1.json','cycle_audit_v1.json']:
        with gzip.open(results/('omega_corr_'+p+'.gz'),'wb') as f:f.write((RUNTIME/p).read_bytes())
    csvout(results/'omega_corr_mutual_cycle.csv',cycle['aggregate'])
    csvout(results/'omega_corr_grouped_row_metrics.csv',ev['summary'])
    csvout(results/'omega_corr_patch_bucket_metrics.csv',review['patch_bucket_groups'])
    write(results/'omega_corr_configuration_index.json',configs)
    paired=[]
    for c in ev['comparisons']:
        for x in c['paired_intervals']:paired.append(dict(scope='joint_strong_controls_common_candidates',omega=c['configuration'],control=x['control'],role='MECHANISM_CHECK',eligible_queries=c['eligible_queries'],**x['interval']))
    for c in review['comparisons']:paired.append(dict(scope='pairwise_common_candidates',**{k:v for k,v in c.items() if k not in ['rows','paired_interval']},**c['paired_interval']))
    csvout(results/'omega_corr_paired_controls.csv',paired)
    # Retain every certified visual-row correction/harm, plus unknown transitions.
    changes=collections.defaultdict(lambda:dict(known_row_corrections=[],known_row_harms=[],uncertified_changed_queries=[]));examples=[];selected_all=[]
    with gzip.open(RUNTIME/'paired_query_results_v1.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            cid=int(r['configuration']);i=int(r['query_index']);changed=r['baseline_track']!=r['chosen_track'];g=changes[str(cid)]
            if r['known_correction']=='True':g['known_row_corrections'].append(i)
            if r['known_harm']=='True':g['known_row_harms'].append(i)
            if changed and (r['baseline_status'] not in ['CORRECT','WRONG'] or r['chosen_status'] not in ['CORRECT','WRONG']):g['uncertified_changed_queries'].append(i)
            if cid==selected_id and r['role']=='MECHANISM_CHECK' and changed:
                item=dict(**r,native_input=qs[i],offline_certification=truth[i],bbox=load_proposal(i,qs));selected_all.append(item)
                if r['known_correction']=='True' or len(examples)<12:examples.append(item)
    write(results/'omega_corr_identity_corrections.json',dict(scope='FROZEN_ROW_RANKING_ONLY_NOT_ACTUAL_SOLVER',selected_layer_fit_only=selected,configurations='omega_corr_configuration_index.json',changes=changes,selected_mechanism_all_changed_rows=selected_all,examples=examples,all_query_results='omega_corr_query_results.csv.gz',unknown_never_forced_negative=True))
    write(results/'omega_corr_solver_decisions.json',dict(status='NOT_EXECUTED',reason='All registered frozen comparisons completed; fixed GateB necessary conditions failed. No calibration training or Omega-Corr solver insertion authorized by scientific gate.',gateB=ev['gateB_necessary_conditions'],actual_new_corr_solver_forks_executed=0,actual_Hungarian_changes=None,actual_threshold_changes=None,actual_new_ID_changes=None,actual_bank_changes=None,actual_final_ID_changes=None,known_actual_ID_corrections=None,known_actual_ID_harms=None,native_context_reconstruction_is_not_corr_solver_execution=True,previous_OmegaF_solver_results_retained_separately=True))
    csvout(results/'omega_corr_full_tracking_metrics.csv',[dict(method=n,status='NOT_EXECUTED',HOTA=None,AssA=None,CVIDF1=None,IDSW=None,DetA=None,IDF1=None,MOTA=None,FPS=None,P95=None,Omega_calls=None,peak_memory_bytes=None) for n in ['GMT','GMT+NativePrior','GMT+OmegaCorr','GMT+DINOcorrespondence','GMT+oldVGGTcorrespondence']])
    runtime=[];receipts=[]
    manifest_by_run={'extract_omega_v1':'extraction/omega/manifest.json','extract_old_vggt_v2':'extraction/old_vggt/manifest.json','extract_dino_v2':'extraction/dino/manifest.json','local_omega_v1':'local_crops/omega/manifest.json','local_dino_v1':'local_crops/dino/manifest.json','native_context_full_v1':'native_context_full_v1/manifest.json'}
    for p in sorted((RUNTIME/'logs').glob('*_receipt.json')):
        receipt=load(p);assert sha256(receipt['log']['path'])==receipt['log']['sha256'];run=p.name.removesuffix('_receipt.json');m=load(RUNTIME/manifest_by_run[run]) if run in manifest_by_run else {}
        runtime.append(dict(run=run,status=receipt['status'],exit_code=receipt['exit_code'],CUDA_VISIBLE_DEVICES=receipt['CUDA_VISIBLE_DEVICES'],child_process_wall_seconds=receipt['child_process_wall_seconds'],GPU_visible_process_wall_hours=receipt['GPU_visible_process_wall_hours'],actual_model_forward_calls=m.get('actual_forward_calls_completed'),actual_image_forward_calls=m.get('actual_image_forward_calls'),peak_allocated_bytes=m.get('CUDA_allocator_peak_bytes'),peak_reserved_bytes=m.get('CUDA_reserved_peak_bytes'),forward_P50=m.get('forward_seconds_P50'),forward_P95=m.get('forward_seconds_P95'),receipt_sha256=sha256(p)))
        receipts.append(dict(path=str(p),sha256=sha256(p),receipt=receipt))
    csvout(results/'omega_corr_runtime.csv',runtime)
    gpu_hours=sum(r['GPU_visible_process_wall_hours'] or 0 for r in runtime);assert gpu_hours<=2
    # All pre-existing files, including GMT core and prior reports, are immutable.
    diff=subprocess.check_output(['git','diff','--name-status','594d9d3237966f1a4d73a3f689eec48e585f0e58','HEAD'],cwd=ROOT,text=True).splitlines();assert all(r.startswith('A\t') for r in diff)
    previous=['docs/'+n for n in ['OMEGA512_FINAL_RESEARCH_REPORT.md','OMEGA512_EXPANSION_REPORT.md','OMEGA512_PILOT_REPORT.md','OMEGA_FINAL_RESEARCH_REPORT.md','TRACKVGGT_FINAL_REPORT.md','TRACKVGGT_CORRECTABLE_ERROR_ANALYSIS.md','TRACKVGGT_DEV12_NATIVE_ASSET_HANDOFF.md']]
    preserved={}
    for p in previous:
        old=subprocess.check_output(['git','show','594d9d32:'+p],cwd=ROOT);assert old==(ROOT/p).read_bytes();preserved[p]=sha256(ROOT/p)
    cache_bytes=sum(p.stat().st_size for p in RUNTIME.rglob('*') if p.is_file() and p.relative_to(RUNTIME).parts[0] not in ['sources','deps','logs']);assert cache_bytes<2*1024**3
    evidence_inputs={str(p):sha256(p) for p in [RUNTIME/'evaluation_v1.json',RUNTIME/'control_review_v1.json',RUNTIME/'cycle_audit_v1.json',RUNTIME/'geometry_support_v1.json',RUNTIME/'native_appearance_v1.json',RUNTIME/'native_prior_canonical2_v1.json',RUNTIME/'candidate_values_v1.json',RUNTIME/'candidate_values_v1.npz',RUNTIME/'reduction_audit_v1.json']}
    audit=dict(status='REGISTERED_FROZEN_STUDY_COMPLETE_GATE_B_FAILED',source=source,states=states,all_preexisting_tracked_files_unchanged=True,previous_report_sha256=preserved,read_only_native_sources_preserved=True,cache_bytes_excluding_sources_deps_logs=cache_bytes,GPU_visible_child_total_hours=gpu_hours,CPU_contract_tests=4,CPU_evaluation_contract_tests=3,hook_parity_Omega='BITWISE_EQUAL_ALL_OUTPUTS',hook_parity_oldVGGT='BITWISE_EQUAL_ALL_OUTPUTS',raw_native_context_max_error=native['max_raw_abs_error'],native_prior_captured_amplitudes_max_error=load(RUNTIME/'native_prior_canonical2_v1.json')['max_captured_amplitude_reconstruction_error'],independent_cycle_checks=cycle['independent_stride257_checks'],evidence_inputs=evidence_inputs,receipts=receipts,source_preservation_limit='GT files were never written by this study; sealed input hashes verified by readers. No claim that an external process cannot modify data.',official416='BLOCKED_WEIGHT_ACCESS',no_new_training=True,no_new_full_tracking=True)
    write(results/'omega_corr_final_audit_v1.json',audit)
    # Human readable tables are generated from measured scalar rows.
    def get(c,role='MECHANISM_CHECK'):
        idx=next(i for i,x in enumerate(configs) if x['config_id']==c)
        return next(r for r in ev['summary'] if r['configuration']==idx and r['group_kind']=='role' and r['group']==role)
    def table(rows):return '| 方法 | 查询 | 有效证据 | 正负可比行 | 正例分数更高 | 已知行修正 | 行破坏 | 净修正 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'+''.join('| '+label+' | '+' | '.join(str(s.get(k,0)) for k in ['queries','with_valid_evidence','discriminable_queries','positive_gap','known_corrections','known_harms','net_known_row_correction'])+' |\n' for label,s in rows)
    grid=[]
    for c in configs:
        if c['layer'] in [11,17,23] or c['backend'].startswith('Native'):
            s=get(c['config_id']);grid.append((c['config_id'],s))
    primary=[(f'Ω L{l} F2 mutual',get(f'FULL_IMAGE_MULTIVIEW|omega|F2|{l}|M0|primary|0')) for l in [11,17,23]]
    for label,n in [('Ω L11独立单视图','FULL_IMAGE_MULTIVIEW|omega|F3|11|M0|primary|0'),('旧VGGT L11','FULL_IMAGE_MULTIVIEW|old_vggt|old_vggt|11|M0|primary|0'),('DINO','FULL_IMAGE_MULTIVIEW|dino|dino|23|M0|primary|0'),('Native1024纯外观','NativeAppearance1024'),('NativeGMT全历史','NativeGMT'),('NativePrior原参数+2','NativePrior2')]:primary.append((label,get(n)))
    selected_stats=get(selected_name);comp=next(c for c in ev['comparisons'] if c['configuration']==selected_name)
    interval_table='| 同候选对照 | Ω减对照R1 | 95%场景/人物聚类区间 |\n|---|---:|---|\n'+''.join(f"| {x['control']} | {x['interval'].get('mean')} | [{x['interval'].get('lower95')}, {x['interval'].get('upper95')}] |\n" for x in comp['paired_intervals'])
    common_text='所有数字为冻结候选行诊断，不是Hungarian、阈值、Bank或最终ID收益。有合法视觉分数时以其最高候选替换原全历史Top1，无分数则保留原Top1；这项激进替换不是已部署的有界融合。NativeGMT/NativePrior保留原全历史聚合。不同方法有效覆盖不同，直接表格用于适用性统计，优劣结论使用共同物理候选集合且有已认证正负例的比较。'
    (docs/'OMEGA_CORR_MATCHING_RESULTS.md').write_text('# Ω-Corr 冻结匹配结果\n\n'+common_text+'\n\n'+table(primary)+'\nFit净行修正：'+str(ev['fit_layer_nets'])+f'；按固定规则选Layer{selected}，没有按机制检查结果换层。机制集384查询中28次原生实际主关联错误，但只有4次全历史行Top1已知错误；这两个分母不可互换。\n\n全部配置的机制检查结果如下；Fit/Scene12、场景、人物、原始像素尺寸和原图patch桶分别存于grouped_row_metrics与patch_bucket_metrics。\n\n'+table(grid)+'\n完整局部余弦矩阵、MNN/单向NN/局部Hungarian/soft cycle、margin和真实位置均保留于哈希绑定的runtime原始文件。硬循环等于mutual覆盖，不算独立发现。\n')
    (docs/'OMEGA_CORR_CONTROL_EXPERIMENTS.md').write_text('# Ω-Corr 强对照与负控制\n\n'+common_text+f'\n\nLayer{selected}的机制检查共同候选包含{comp["eligible_queries"]}条可认证正负查询；不是仅取共同查询而保留不同候选。Ω独有且修正原生全历史错误的查询数为{len(ev["selected_mechanism"]["unique_repairs"])}。\n\n'+interval_table+'\n所有Layer11/17/23、F0/F1/F2/F3、M0/M1/M2、N1/N2、ROI rawmean/ROIAlign、随机局部配对、mutual/cycle，以及224/336同像素crop的比较已完成，详见paired_controls.csv及压缩control_review全部明细。原始13个可靠F图像对仅用于B10辅助，不能选择性对比全部160；44,156条F2/M0候选记录中只有188条有缓存F，180条同时有多点匹配，缺失F为未知。极线相容不证明同一人体3D表面。\n\nN1为人工[A,A]/[B,B]错误搭档安慰剂，保持原始slot，不能称新的物理相机。F3单图总用first-slot，故N1用于识别slot/partner混杂。N2固定旋转真实候选到区域映射，候选ID和GT保持。\n\n原图Ω输入384×688、patch16；旧VGGT/DINO518×518、patch14、有效区排除padding。像素预算接近但目标尺度不同，故额外使用共同候选和224/336相同裁剪。局部放大不增加原图真实细节、不是新相机。旧VGGT原forward包含world_points，不将其与Ω延迟差当架构结论。Native1024是纯ROI外观，1152包含128维空间邻域特征。NativePrior正式参数为weight2/cap2/age20；.1/.3只是附加幅度对照，重建误差0。\n')
    (docs/'OMEGA_CORR_ACTUAL_SOLVER_RESULTS.md').write_text('# Ω-Corr 原生求解状态\n\nNOT_EXECUTED：固定GateB未通过，因此没有训练校准头，也没有将Ω-Corr插入get_asso或执行新状态分叉。Hungarian、strict threshold、新ID、Bank与最终身份变化均为null/未测，不能写成零收益。\n\n160组原生上下文重建仅校验OFF排序和既有实际匹配分数，不是Ω-Corr真实求解实验。原生缓存重建涉及8,456次新增图像forward、242张预检复用和552张封存Scene12上下文复用；309次主关联调用，共1,895查询，旧缓存raw矩阵与实际已选分数误差0。原来的Ω-F求解负结果单独保留，不能作为本轮失败证据。\n\n行排序净纠错为负且没有独有证据，不绕过门槛运行新solver/fullvideo。机器状态见omega_corr_solver_decisions.json。\n')
    omega=load(RUNTIME/'extraction/omega/manifest.json');joint=[p['forward_seconds'][0] for p in omega['pair_receipts']]
    (docs/'OMEGA_CORR_EFFICIENCY_REPORT.md').write_text(f'# Ω-Corr 实测成本与条件未执行\n\n所有已运行GPU可见子进程（含失败、导入、读权重、预处理、推理与导出）的累计墙时为{gpu_hours:.6f} GPU-visible process-hours；不是CUDA纯kernel时间，也不是GPU利用率积分。第一阶段2小时预算通过。研究缓存{cache_bytes} bytes（不含代码、deps及日志）小于2GiB。\n\n完整Ω研究提取800次forward=160联合+320真正单视图+320错搭档控制。另接口5次；旧VGGT成功160+hook校验3，DINO160次双图独立批次；224/336局部crop每个模型686次独立proposal调用。首次old/DINO patch16误用失败成本也保留，源代码显示各完成首个forward后出错，计数器未导出，不能冒充实测计数。\n\nΩ联合forward的160组P50={np.quantile(joint,.5):.6f}s、P95={np.quantile(joint,.95):.6f}s；包括首次冷forward、不含权重加载。全子进程成本、各模型forward P50/P95、峰值allocated/reserved分别见runtime.csv和原始manifest/receipts。完整研究提取含对照额外调用，不等于默认部署每组一次调用。\n\nFULL_TRACKING_EFFICIENCY=NOT_EXECUTED/NOT_MEASURED。没有GMT原生在线FPS、系统FPS、在线P95、跳过率或10%/20%触发收益；不以缓存读取/离线提取速度冒充实时系统。局部686次crop调用只作有限尺度机制对照，未实现在线调度。模型常驻内存与系统共存开销未在线测量；独立提取峰值不等于系统额外显存。\n')
    (docs/'OMEGA_CORR_REVIEWER_AUDIT.md').write_text('# Ω-Corr 审稿核验\n\n- 官方Ω512严格1411键、SHA已核验，416仍BLOCKED_WEIGHT_ACCESS；不冒充416复现。\n- Ω与旧VGGT hook全部官方输出bitwise相等；真实4层2048维，Ω排除17特殊token、patch16；F0不是纯单视图。\n- 0/1真实patch不复制，M0/M1/M2独立，padding排除；CPU坐标/缺失/分母4项和离线标签/同候选/聚类3项测试通过。\n- 原生1152维描述符与纯1024外观分开；保留正式NativePrior2，没有削弱基线；raw/RNG/排序/实际匹配分数重建误差0。\n- 原160组在feature前封存；96旧组只有相机/对应缓存，故重提token合理。56追加错误组与8诊断组均按feature前规则固定。local32预封存索引修复公开，主160未变。部分详细reduction规则在提取开始后、读取分数前固定，不宣称完全盲预注册。\n- Fit/机制/诊断角色不变，所有均GMT TRAIN-seen，Scene12已接触标签；没有DEV5/TEST调参或新泛化声明。\n- 真实输入/离线GT标签分开，mixed/unknown不强作负类；无同步候选纳入全1895查询。\n- 403提取配置全部归约，加上原生及几何辅助配置完整公开；不是只跑一个层或只选择可靠F。\n- 已知行修正和破坏不等于原实际主关联错误修复。强行替换视觉Top1很差，也不证明所有可能有界融合都失败；只是本注册路线未获得进入训练的独有证据。\n- 274550个有效矩阵MNN全量核验，1068个固定抽样独立softcycle trace核验，最大误差约1.58e-7。循环自洽不等于3D真值。\n- 原有源码、GMT核心、GT、审计资产与7份旧报告保留；final_audit记录报告SHA、已读资产与raw输出SHA。所有相对parent修改均为新增文件。\n- 后续训练/实际求解/完整视频按门槛未执行，指标null/NOT_EXECUTED；没有伪造全视频0增益。\n\n数值与来源以final_audit、experiment_manifest和每个测量receipt为准。原始大矩阵留在隔离runtime并哈希绑定，不推送原图或大于100MiB原始文件；已发布lossless候选标量NPZ供独立排序复算。\n')
    (docs/'OMEGA_CORR_FINAL_RESEARCH_REPORT.md').write_text(f'# Ω-Corr 最终研究报告\n\n本轮完成160组、冻结Ω512/旧VGGT/DINO/原生GMT的全部注册独立机制对照，未达到独有跨视角证据GateB。按固定门槛停止新增训练、Ω-Corr solver插入和完整视频评测。结论只适用于该数据、层、局部匹配定义与有限预算，不否定一切Ω身份用途，也不援引旧Ω-F负结果作为证明。\n\n'+ '\n'.join(f'- `{k}`：`{v}`' for k,v in states.items())+'\n\n'+table(primary)+f'\n仅Fit选择Layer{selected}：净行修正{ev["fit_layer_nets"]}。机制检查有{selected_stats.get("known_corrections",0)}次已知行修正、{selected_stats.get("known_harms",0)}次破坏，独有修正0。该强行视觉排序结果不能直接代表有界融合或实际ID效果。\n\n'+interval_table+'''\n研究问题逐项回答：\n\n1. 联合视角是否优于单视图：共同候选聚类区间未建立稳健优势，不能声称独有跨视角交互收益。\n2. 层价值：11/17/23全部完成，Fit选11；三个层均未跨过门槛，不按机制检查挑更好层。\n3. 覆盖：1972物理proposal中916个（46.45%）少于2真实M0 patch；局部多点不可用，保留ROI/裁剪对照。\n4. 人物与背景：M0/M1/M2均单列；同候选目标对背景环的区间见paired_controls.csv，不将背景共现当人物几何。\n5. Mutual与ROI：raw均值和ROIAlign均完成，同候选表没有支撑本注册独有身份假设；不能靠不同可用分母宣布优越。\n6. Cycle：软循环是附加数值，硬循环与mutual覆盖等价；没有满足门槛的额外真实身份纠正，实际求解未测。\n7. 对旧VGGT/DINO：各层修正、破坏和净差见上表/完整matching报告，共同有效97条机制正负查询的区间见上。\n8. 独有修正：Fit选层的共同候选、对强对照独有且修正原全历史Top1错误为0。其余固定层/cycle/local结果全部保留，不能事后择优提升为主结论。\n9. 适用性：1895查询中992无同步跨相机参考，1383无可认证同步纯正例；原132主关联错误中103有同步纯正例。原完整TRAIN的169468/1553/650/314/297分母保留，不把有限样本当总体。\n10. 已知修正/破坏：本报告数字为冻结行排序，原实际384机制查询有28主关联错误，但仅4条全历史Top1已知错误。二者不能混算。\n11. GTA/Hungarian：没有新Ω-Corr状态分叉，变化数未测。OFF上下文重建用于严谨原生对照，不能冒充新证据进入求解。\n12. 最终ID：NOT_EXECUTED，不能写0变化或0增益。\n13. 调用预算与系统延迟：10%/20%不确定性在线触发未执行，系统FPS/P95未测；离线子进程和forward成本另见效率报告。\n14. 为什么不直接NativePrior/DINO：本轮未提供需要更大Ω模型的证据。NativePrior原参数2/cap2与纯1024 GMT外观在同候选上是必须保留的强基线；本轮不证明它们完整视频一定提升。\n15. WWW独立贡献差距：没有可复现的Ω独有跨视角净纠错，没有原生solver最终身份净收益，没有完整视频精度—在线成本证据；不能以工程实现充当论文贡献。\n\n小目标补救已执行：固定32组224/336同像素crop，Ω/DINO各686次真实独立forward。机制子集73查询、26条有可认证共同正负；Ω没有稳健优于同crop DINO，更多编码patch不等于新增原始细节。所有主配置、安慰剂、失败及未知保留。\n\n复现：使用final_audit内每个receipt记录的精确source_commit、command、环境和输入SHA，在新的输出目录执行对应脚本；不要覆盖原实验。冻结原始候选分数、共同候选索引、逐query输出已发布。README式细节见SOURCE_AUDIT、TOKEN_INTERFACE、DATA_AND_COVERAGE、MATCHING_RESULTS、CONTROL_EXPERIMENTS、ACTUAL_SOLVER_RESULTS、EFFICIENCY_REPORT、REVIEWER_AUDIT。\n''')
    output_files=[p for p in results.glob('omega_corr_*') if p.is_file() and p.name!='omega_corr_experiment_manifest.json']
    write(results/'omega_corr_experiment_manifest.json',dict(status='COMPLETE_REGISTERED_FROZEN_STUDY_NEGATIVE_GATE_B',source=source,**states,selected_layer_fit_only=selected,fit_layer_nets=ev['fit_layer_nets'],gateB_necessary_conditions=ev['gateB_necessary_conditions'],training='NOT_EXECUTED_GATE_B_FAILED',new_solver_forks='NOT_EXECUTED_GATE_B_FAILED',full_tracking='NOT_EXECUTED_GATE_C_NOT_ESTABLISHED',official416='BLOCKED_WEIGHT_ACCESS',parent_geometry_results='RETAINED_NOT_EVIDENCE_OF_CORR_FAILURE',all_registered_frozen_comparisons_completed=True,artifacts={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in output_files},raw_runtime_inputs=evidence_inputs,cache_bytes=cache_bytes,GPU_visible_child_hours=gpu_hours))
    print('PUBLISHED',states,'GPUhours',gpu_hours,'cache bytes',cache_bytes,flush=True)

_PROPOSALS=None
def load_proposal(i,qs):
    global _PROPOSALS
    if _PROPOSALS is None:_PROPOSALS=json.loads((RUNTIME/'native_candidates_v1.json').read_text())['proposals']
    return _PROPOSALS[':'.join(map(str,qs[i]['query']))]['bbox_xyxy']

if __name__=='__main__':main()

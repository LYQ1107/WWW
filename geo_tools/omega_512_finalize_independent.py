"""Close the registered independent512 finite study only on a failed identity admission."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from omega_common import ROOT,RUNTIME,DATA,sha256,save_new


def main():
    identity=json.loads((ROOT/'results/omega512_identity_counterfactual_v1.json').read_text())
    verification=json.loads((ROOT/'results/omega512_solver_verification_v1.json').read_text())
    assert verification['status']=='PASS'
    assert identity['status']=='NO_CERTIFIED_MULTISCENE_NET_FINAL_IDENTITY_GAIN'
    assert not any(r['passes_online_admission'] for r in identity['summary'])
    pilot=json.loads((ROOT/'results/omega512_independent_evidence_v1.json').read_text())
    expansion=json.loads((ROOT/'results/omega512_expansion_evidence_v1.json').read_text())
    scope=json.loads((ROOT/'results/omega512_identity_scope_v1.json').read_text())
    assert expansion['manifest']['uniform_admission']['passes_pilot_gate']
    # Preserve every small fork matrix/context/actual solver record for external review.
    raw=Path(identity['underlying_output']['path']);assert sha256(raw)==identity['underlying_output']['sha256']
    dest=ROOT/'results/omega512_solver_numeric_v1.jsonl.gz'
    with dest.open('xb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(raw.read_bytes())
    old=json.loads((ROOT/'results/omega_handoff_index_v1.json').read_text())
    for p,h in old['files'].items():assert sha256(ROOT/p)==h,p
    gthashes={}
    for row in json.loads((ROOT/'results/train_gt_quality_audit.json').read_text())['rows']:
        path=DATA/'train'/f"{row['scene']}_View{row['view_id']}"/'gt/gt.txt'
        assert sha256(path)==row['raw_sha256'];gthashes[str(path)]=row['raw_sha256']
    core=subprocess.check_output(['git','diff','--name-only','c6999a7ab1d22549816b5219734f53c8dcb74178','--','gtr/modeling','gtr/tracking'],cwd=ROOT,text=True)
    assert not core.strip()
    prior=Path('/data1/liuyeqiang/WWW_trackvggt_20261015')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=prior,text=True).strip()=='c6999a7ab1d22549816b5219734f53c8dcb74178'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=prior,text=True).strip()
    allocation=RUNTIME/'omega512_independent_v1'
    dirs=[allocation,RUNTIME/'pose_audits/omega512_remaining72_v1',RUNTIME/'pose_audits/old_vggt_remaining72_v1']
    cache_bytes=sum(p.stat().st_size for d in dirs for p in d.rglob('*') if p.is_file())
    assert cache_bytes<2147483648
    lines=['# 独立512有限研究：最终证据与GMT准入判定','',
        '**本轮固定方案未通过身份净纠错准入；不接入生产GMT。** 这不是416复现，也不构成Ω在所有场景无效的结论。','',
        '使用已核验512原版PyTorch权重，SHA256 `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`，官方balanced预处理。416继续单独NOT_EXECUTED，本轮没有等待或替代它。','',
        '先推送有限预注册57483f93，再执行24组C1/C2匹配恢复；C2在3组、2场景通过AB/BA，随后按预注册扩展其余72组。新增42均匀组中6组通过，场景8、9；困难30组不计入该准入。全部96组和三项匹配配置保留。所有背景阈值未变；没有逐样本选最佳配置。','',
        'C2总计13/96组在Ω与旧VGGT中共同通过，而且是相同图像对。可靠支持局限于两个场景，不能直接声称Ω独立超越旧VGGT。C1两者均0/96。原24组模型结果没有重跑。','',
        f"几何准入后复用封存的原生身份审计：固定96组中{scope['fixed_cohort_known_primary_errors']}个已知primary错误，有{scope['eligible_certified_correction_opportunities']}个可靠图像对上的合法同步纯正例纠错机会，涉及2场景、3个按场景区分的人物。这些是机会计数，不是纠错数量。",'',
        '有限求解审计在全部13个可靠图像对及2个固定不可靠控制上进行，15组×11分支。使用原生OFF连续前缀，各当前组独立分叉；恢复随机状态、ID全局集合及Bank状态，共享相同当前检测前向，分支状态随后丢弃。没有将几何接入生产在线流程，没有向后续视频传播反事实历史。','',
        '15组均通过原生封存输出与零残差分支的精确一致性检查，包括真实求解、Bank和最终提交ID。G0也代表各配置lambda=0的相同对象返回；其余固定lambda为.1和.3。G4深度未校准，NOT_EXECUTED。','',
        '|分支|lambda|primary纠正/破坏|最终认证纠正/破坏|最终ID变化|认证人物/场景|准入|',
        '|---|---:|---:|---:|---:|---|---|']
    for r in identity['summary']:
        lines.append(f"|{r['arm']}|{r['weight']}|{r['primary_corrections']}/{r['primary_harms']}|{r['known_final_corrections']}/{r['known_final_harms']}|{r['final_ID_changes']}|{len(r['corrected_scene_qualified_people'])}/{len(r['corrected_scenes'])}|{r['passes_online_admission']}|")
    lines+=['', 'G5仅在当前合法对侧候选内旋转几何分数；只有一个候选的行保持不变，单例与多候选行计数均保留，不能视作这些行上的强安慰剂。NativePrior采用已有20组年龄规则，几何仅作用同步候选，适用范围差异单独保留。','',
        '完整结果同时区分原始候选argmax、激活后轨迹排序、实际primary分配、Bank调用/接受、最终提交ID，以及未知比较。排序改变不计作纠错。已知计数为实测；未知或混合历史保留为未知，不用新ID数值对齐GT。新轨迹仅在存在已认证的历史纯正例和合法纯正例时判为已知碎片化。','',
        'GT仅在事后认证中使用封存标签；初始参考标签只可从原封存“纯正例”证书恢复，不进行新GT匹配。细化认证规则300be79d在推理启动后、查看求解结果之前固定；本研究是已接触TRAIN标签的开发诊断，不能称完整盲预注册验证。','',
        f"独立数值检查重算24组240项、72组1080项背景指标；又核验{verification['primary_calls']}次真实primary调用的null-softmax/轨迹聚合、Hungarian和严格阈值，{verification['queries']}条查询，观察{verification['memory_calls_observed']}次原生Bank求解。原始小矩阵、来源、分支输出和失败/未知案例均发布。",'',
        f"本轮新增GPU可见完整子进程wall合计{identity['combined_new_GPU_visible_process_wall_seconds']:.3f}秒，即{identity['combined_new_GPU_visible_process_wall_seconds']/3600:.6f}小时，低于.5小时上限；这包括相机模型新增各144次前向及GMT反事实前缀/分支。原24组历史GPU诊断成本另计，不重复计作新增。新增研究缓存{cache_bytes}字节，低于2GiB（环境和权重单独核算）。",'',
        '没有训练Ω/GMT，没有修改GMT主干、GTA、Hungarian、历史或ID分配实现；仅隔离研究进程在logit处构造预注册残差。既有TrackVGGT工作树、46项历史封存证据和原始训练GT哈希核验未变。DEV5/TEST未参与调参。','',
        '**后续状态**：生产在线融合、完整视频跟踪B0–B6和实时效率验收均NOT_EXECUTED；系统FPS、HOTA、MOTA、IDF1等均NOT_MEASURED。没有通过净纠错证据门槛，因此不启动这些条件分支。需要新的研究假设时先另行注册，不能继续用本轮结果调阈值或挑案例。','',
        '权威产物：results/omega512_identity_counterfactual_v1.json（所有查询及认证）；results/omega512_solver_numeric_v1.jsonl.gz（完整小矩阵和真实求解记录）；results/omega512_solver_verification_v1.json（独立重算）；results/omega512_independent_evidence_v1.json与omega512_expansion_evidence_v1.json（全部几何与成本）；results/omega512_handoff_v1.json（当前独立512索引）。']
    report=ROOT/'docs/OMEGA512_FINAL_RESEARCH_REPORT.md';report.write_text('\n'.join(lines)+'\n')
    save_new(ROOT/'results/omega512_completion_audit_v1.json',dict(status='FINITE_REGISTERED512_STUDY_COMPLETE_NO_ONLINE_ADMISSION',
        preregistration_before_new_matching=True,conditional72_executed_only_after_pilot_gate=True,
        uniform_geometry_gate_passed=True,identity_gate_passed=False,production_GMT_integration=False,
        original_GMT_core_unchanged=True,old_trackvggt_worktree_clean=True,historical_sealed_files_verified=len(old['files']),
        original_GT_hashes=gthashes,all15_native_and_zero_fork_parities=True,solver_verification=verification,
        finite_budget_GPU_seconds=identity['combined_new_GPU_visible_process_wall_seconds'],finite_budget_GPU_seconds_limit=1800,
        research_cache_bytes=cache_bytes,research_cache_limit_bytes=2147483648,
        no_general_universal_negative_claim=True,official416='SEPARATE_NOT_EXECUTED',
        conditional_online_and_fullvideo='NOT_EXECUTED_IDENTITY_ADMISSION_FAILED',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    files={str(p.relative_to(ROOT)):sha256(p) for pattern in ['docs/OMEGA512*.md','manifests/omega512*.json','results/omega512*','geo_tools/omega_512*.py','geo_tools/test_omega_512*.py'] for p in ROOT.glob(pattern) if p.is_file() and p.name!='omega512_handoff_v1.json'}
    save_new(ROOT/'results/omega512_handoff_v1.json',dict(status='FINITE512_COMPLETE_NO_IDENTITY_ADMISSION',
        research_role='INDEPENDENT_512_NOT_416_REPRODUCTION',files=files,
        raw_solver_numeric=dict(path=str(dest.relative_to(ROOT)),sha256=sha256(dest),bytes=dest.stat().st_size)))
    print('finite512 closed: identity admission failed; productionGMT unchanged; files',len(files))

if __name__=='__main__':main()

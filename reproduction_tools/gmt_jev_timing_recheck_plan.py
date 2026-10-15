"""CPU-only preparation of an unapproved timing recheck; never launches jobs."""
from gmt_jev_common import *
from gmt_jev_job import spent


def main():
    registry = read(REPORTS / 'EXPERIMENT_REGISTRY.json')['jobs']
    kinds = ['original', 'mlp_learned', 'mlp_learned_cached']
    names = ['P5_original_full_image_GPU7_retry',
             'P5_mlp_learned_full_GPU7', 'P5_mlp_learned_cached_full_GPU7']
    references = []
    for kind, name in zip(kinds, names):
        job = registry[name]
        speed_path = REPORTS / f'SPEED_{kind}.json'
        result = read(speed_path)
        assert job['status'] == result['status'] == 'COMPLETE'
        assert len(result['trials']) == 3 and job['GPU'] == 7
        references.append(dict(kind=kind, job=name, whole_job_seconds=job['seconds'],
                               speed=ref(speed_path), log=job['log']))
    order = [kinds[i:] + kinds[:i] for i in range(3)]
    assert all(sorted(block) == sorted(kinds) for block in order)
    assert all(len({block[position] for block in order}) == 3 for position in range(3))
    estimate = 1.10 * sum(r['whole_job_seconds'] for r in references) + 10
    current_remaining = (12 - spent()) * 3600
    assert estimate > current_remaining, 'Reassess whether approval is still needed'
    plan = dict(
        status='PROPOSED_NOT_AUTHORIZED_NOT_RUN',
        scientific_success=False,
        purpose='Separate contemporaneous full-image efficiency from serial host timing drift; cannot by itself prove scientific success.',
        source_HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        scope=dict(video=16, development_only=True, new_training=False,
                   new_download=False, new_DEV=False, unseal=False, official_TEST=False),
        fixed_order=order, complete_runs=9, fixed_seed=SEED, physical_GPU=7,
        references=references,
        resource_forecast=dict(estimated_seconds=estimate, estimated_GPU_hours=estimate/3600,
            formula='sum(three related whole 3-trial jobs)*1.10 +10 seconds',
            current_GPU_hours=spent(), current_12h_remaining_seconds=current_remaining,
            requested_additional_GPU_hours_max=2, requested_cumulative_GPU_hours_max=14,
            current_budget_unchanged=True, new_retained_bytes_max=16*2**20,
            overall_retained_bytes_max=512*2**20, destination_free_GiB_min=30),
        freeze=dict(GMT_checkpoint_SHA256=STAGE2_SHA,
            trained_checkpoint=read(REPORTS/'ONLINE_mlp_learned.json')['trained_checkpoint'],
            exact_cache_proof=ref(REPORTS/'LEARNED_MLP_FEATURE_CACHE_PARITY.json'),
            source={str(p.relative_to(ROOT)): sha(p) for p in [
                ROOT/'gtr/modeling/gmt_jev/correction.py',
                ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py',
                ROOT/'reproduction_tools/gmt_jev_runtime.py']}),
        measurement_contract=[
            'Same single model object; no gradients; reattach only the frozen relevant actor and committed-state observer.',
            'Before each complete run: reset all native state, actor summaries and seed; same eight-frame warmup excluded from timing.',
            'Include real image IO/resize, detector, VFCE/RPCE, GTA, native solver, Gallery/Bank and stock final processing; same synchronized Stage2 instrumentation.',
            'Record timestamps, scene/image FPS, Stage2 p50/p95, peak memory and per-run stock fingerprint; compare each fingerprint exactly to its existing frozen method reference.',
            'Verify the chosen GPU has no other compute PID before and after each trial. Disclose remaining shared-host variability.',
            'No omission/retry of slow trials, no stopping for favorable FPS and no selection of repetition counts; retain partial results on failure or resource termination.',
            'Report all nine runs, within-block ratios and the ratio of three-run arithmetic mean FPS values; preregister 0.95 efficiency target.',
            'Keep all original block-timed results as primary historical measurements; label new interleaved results a prospective robustness recheck, never silently replace a failed result.',
            'No HOTA/seed/independence claims from timing. Quality reuse requires exact frozen prediction equality.',
            'Authorization must arrive before GPU launch; the existing 12h executor guard stays unchanged until then.'
        ],
        unresolved=['Efficiency has failed in existing measurements',
                    'MCMT evidence contribution and causal error propagation unproven',
                    'Additional independent training seeds not run',
                    'No qualified independent complete video established within current authorized scope'])
    save(REPORTS/'TIMING_RECHECK_PROPOSAL.json', plan)
    rows = [' | '.join(block) for block in order]
    text = f'''# 下一轮同卡交错计时提案（尚未授权、未启动）

现有三轮完整图像FPS：GMT 3.409768、默认MLP 3.129765、缓存MLP 2.848982。默认与缓存均未通过95%；缓存严格等价，不证明加速。原计时是串行方法块，尚不能分离宿主机时间漂移与模块成本。

只建议先复核效率，不扩大CIC，不重新训练，也不把本项当作完整科研目标。详细输入SHA、冻结checkpoint、原始记录与执行合同见 `reports/GMT_JEV/TIMING_RECHECK_PROPOSAL.json`。

固定三块顺序如下，每项均为TRAIN16整段1034场景、2068图像的冷启动完整推理；每种方法恰好处于每个顺序位置一次：

| 块 | 第1次 | 第2次 | 第3次 |
|---|---|---|---|
| 1 | {rows[0]} |
| 2 | {rows[1]} |
| 3 | {rows[2]} |

冻结同一完整GMT和300步MLP权重、合法候选、原生求解器、提交规则、残差与缓存策略、seed20261009。复用一个模型对象，每次重置原生及派生状态、执行相同8帧warmup后计时；不得复用perception cache。计时包含真实图像加载到最终stock处理，统一Stage2同步剖析。逐次核对预测指纹，不一致就保留失败，不能复用质量结论。

保存全部9次FPS、块内比值、三次算术均值的比值、时间戳、Stage2分位数、显存和输出指纹。95%门槛固定，不挑有利轮次、不按快慢重跑。旧结果继续保留；新结果单列为前瞻效率稳健性复核。观测区间不冒充置信区间或训练种子验证。空闲GPU7优先，检查同卡占用，不结束其他任务；宿主机剩余负载变化仍需披露。

相关完整作业耗时之和乘1.10加10秒，预测 **{estimate:.2f}秒（{estimate/3600:.3f} GPU-hours）**，包括模型加载、warmup和输出处理等已计入旧作业的成本。本轮12小时剩余仅 **{current_remaining:.2f}秒**，无法完成。提请追加最多2 GPU-hours，并将累计上限明确改为最多14小时；执行时还需同时限制本次最多2小时。新增留存最多16MiB、整个研究最多512MiB、home至少30GiB余量不变。审批前原12小时守卫不变，本文件不会启动任何作业。

即使复核通过效率，仍缺训练种子、MCMT证据消融、真正因果传播时间和合格独立完整视频。当前TRAIN16已暴露；本提案不授权新数据、DEV17–19、封存20–22、官方TEST或Full24。任何后续科学成功仍需原Goal的全部条件。
'''
    (DOCS/'TIMING_RECHECK_PROPOSAL_ZH.md').write_text(text)
    print('PROPOSAL_ONLY_NO_GPU', round(estimate/3600, 6), 'hours;', round(current_remaining, 2), 'seconds remaining')


if __name__ == '__main__':
    main()

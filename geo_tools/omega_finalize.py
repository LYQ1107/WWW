"""Publish completed bounded diagnostics while explicitly leaving416 incomplete.

This entry cannot declare scientific success or admit GMT online experiments.
Every missing conditional result is a typed NOT_EXECUTED/NOT_MEASURED value.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from omega_common import ROOT, RUNTIME, provenance, save_new, sha256


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostic-only', action='store_true', required=True)
    args = parser.parse_args()
    background_path = ROOT/'results/omega_distinct_heldout_comparison_v1.json'
    background = json.loads(background_path.read_text())
    assert background['status'] == 'COMPLETE_COMMON_DISTINCT_HELDOUT_DIAGNOSTIC'
    manifests, runs, clocks = {}, {}, {}
    receipts = []
    for method, tag in [('omega512', 'omega512_initial24_v1'), ('old_vggt', 'old_vggt_initial24_v1')]:
        path = RUNTIME/'pose_audits'/tag/'manifest.json'
        assert sha256(path) == next(s['sha256'] for s in background['source_manifests'] if s['path'] == str(path))
        manifest = json.loads(path.read_text())
        assert manifest['pairs'] == 24 and manifest['actual_model_calls_completed'] == 48
        assert sha256(manifest['outputs']['path']) == manifest['outputs']['sha256']
        clock_path = RUNTIME/'logs'/(tag+'_receipt.json')
        clock = json.loads(clock_path.read_text())
        assert clock['status'] == 'CHILD_FINISHED' and clock['exit_code'] == 0
        assert sha256(clock['log']['path']) == clock['log']['sha256']
        assert clock['command'][-1] == ('24' if method == 'omega512' else tag)
        runs[method] = [json.loads(line) for line in Path(manifest['outputs']['path']).read_text().splitlines()]
        assert all(o['status'] == 'FINITE_INTERFACE_OUTPUTS' for r in runs[method] for o in r['outputs'].values())
        manifests[method], clocks[method] = manifest, clock
        receipts.extend([dict(path=str(path), sha256=sha256(path)), dict(path=str(clock_path), sha256=sha256(clock_path))])
    acquired_path = ROOT/'results/omega_public_512_acquisition_v1.json'
    acquired = json.loads(acquired_path.read_text())
    assert sha256(acquired['path']) == acquired['sha256'] == manifests['omega512']['checkpoint']['sha256']
    assert manifests['omega512']['resolution'] == 512
    assert manifests['omega512']['model_strict_state_keys'] == 1411
    assert all(r['role'] == 'SEPARATE_512_NATURAL_DIAGNOSTIC' for r in runs['omega512'])
    source = provenance()
    # Prove code scope separately from an unexecuted new whole-GMT parity run.
    core = ['gtr/modeling/meta_arch/gtr_rcnn.py', 'gtr/config.py', 'gtr/geometry/runtime.py',
            'gtr/geometry/pair_score_adapter.py']
    unchanged = {}
    for name in core:
        baseline = subprocess.check_output(['git', 'show', 'c6999a7a:'+name], cwd=ROOT)
        assert (ROOT/name).read_bytes() == baseline
        unchanged[name] = sha256(ROOT/name)
    targets = ['docs/OMEGA_PHASE0_INTERFACE_AUDIT.md', 'docs/OMEGA_CAMERA_GEOMETRY_REPORT.md',
               'docs/OMEGA_BACKGROUND_VALIDITY_REPORT.md', 'docs/OMEGA_IDENTITY_CORRECTION_REPORT.md',
               'docs/OMEGA_ONLINE_TRACKING_REPORT.md', 'docs/OMEGA_FINAL_RESEARCH_REPORT.md',
               'results/omega_camera_pose_comparison.csv', 'results/omega_identity_candidate_scores.csv',
               'results/omega_identity_correction_audit.json', 'results/omega_runtime.csv',
               'results/omega_full_tracking_metrics.csv', 'results/omega_experiment_manifest.json']
    if any((ROOT/p).exists() for p in targets):
        raise ValueError('refuse to overwrite a prior published report')
    pose_rows, runtime = [], []
    for method, rows in runs.items():
        for row in rows:
            for order, output in row['outputs'].items():
                pose_rows.append(dict(pair_id=row['pair_id'], scene=row['scene'], group=row['group'], method=method,
                  order=order, finite_interface=output['status'], input_shape=json.dumps(output['shapes']['images']),
                  pose_shape=json.dumps(output['shapes']['pose_enc']), depth_shape=json.dumps(output['shapes']['depth']),
                  confidence_shape=json.dumps(output['shapes']['depth_conf']),
                  preprocess_and_forward_seconds=output['preprocess_and_forward_seconds'],
                  F_swap_difference=row.get('order', {}).get('F_transpose_scale_sign_error', row.get('F_swap_difference', 'NOT_MEASURED')),
                  camera_GT_pose_error='NOT_MEASURED'))
        manifest, clock = manifests[method], clocks[method]
        times = [o['preprocess_and_forward_seconds'] for r in rows for o in r['outputs'].values()]
        runtime.append(dict(method=method, status='MEASURED_BOUNDED_GEOMETRY_DIAGNOSTIC', pairs=24,
                  model_calls_attempted=manifest['actual_model_calls_attempted'], model_calls_completed=48,
                  full_child_seconds=clock['child_process_wall_seconds'],
                  GPU_visible_process_wall_hours=clock['GPU_visible_process_wall_hours'],
                  preprocess_forward_check_p50_seconds=float(np.median(times)),
                  preprocess_forward_check_p95_seconds=float(np.quantile(times, .95)),
                  CUDA_allocator_peak_bytes=manifest['CUDA_allocator_peak_bytes'],
                  CUDA_reserved_peak_bytes=manifest['CUDA_reserved_peak_bytes'],
                  system_tracking_FPS='NOT_MEASURED', whole_device_lifetime_peak='NOT_MEASURED',
                  scope='AB/BA repeated-image geometry calls; no GMT; old forward also emits world points; SIFT/fit/export charged in Omega producer, reused by old control'))
    identity = dict(status='NOT_EXECUTED', reason='MAIN416_WEIGHT_ACCESS_BLOCKED_AND_NO_ADMITTED_IDENTITY_GEOMETRY',
                    pure_geometry_identity_evaluation='NOT_EXECUTED', association_matrix_change='NOT_EXECUTED',
                    actual_Hungarian_Bank_final_ID_correction='NOT_EXECUTED', known_corrections='NOT_MEASURED',
                    known_harms='NOT_MEASURED', net_identity_gain='NOT_MEASURED',
                    native_scope_source=dict(path=str(ROOT/'results/omega_native_synchronous_scope_v1.json'), sha256=sha256(ROOT/'results/omega_native_synchronous_scope_v1.json')),
                    native_error_counts_are_not_method_corrections=True,
                    candidate_CSV_rows='six explicit unexecuted-condition placeholders, no evaluated candidate events')
    condition_rows = [dict(condition='G'+str(i), status='NOT_EXECUTED', query_key='NOT_EXECUTED',
                          geometry_score='NOT_MEASURED', correct_negative_margin='NOT_MEASURED',
                          actual_solver_correction='NOT_MEASURED') for i in range(6)]
    tracking = [dict(method='B'+str(i), status='NOT_EXECUTED', HOTA='NOT_MEASURED', AssA='NOT_MEASURED',
                    CVIDF1='NOT_MEASURED', IDSW='NOT_MEASURED', DetA='NOT_MEASURED', MOTA='NOT_MEASURED',
                    CVMA='NOT_MEASURED', FPS='NOT_MEASURED', reason='PRECEDING_MAIN_GEOMETRY_IDENTITY_GATES_NOT_SATISFIED') for i in range(7)]
    manifest = dict(schema='omega-experiment-index/v1', status='DIAGNOSTICS_COMPLETE_MAIN416_ACCESS_BLOCKED',
                    scientific_goal_complete=False, engineering_preparation_complete=True,
                    main_416='NOT_EXECUTED_BLOCKED_WEIGHT_ACCESS', actual_512='BOUNDED_DIAGNOSTIC_COMPLETE',
                    OMEGA_GEOMETRY_SUCCESS='NOT_ESTABLISHED_FOR_MAIN416',
                    OMEGA_IDENTITY_EVIDENCE_SUCCESS='NOT_MEASURED', OMEGA_TRACKING_SUCCESS='NOT_MEASURED',
                    OMEGA_EFFICIENCY_SUCCESS='NOT_MEASURED',
                    diagnosis='WEIGHT_ACCESS_BLOCKED_MAIN416; bounded512 comparison separately reported, not a substitute',
                    expansion_to_96='NOT_EXECUTED', depth_identity_assistance='NOT_EXECUTED',
                    offline_identity_probe='NOT_EXECUTED', online_fusion='NOT_EXECUTED', full_video_evaluation='NOT_EXECUTED',
                    native_GMT_core_unchanged_by_bytes=unchanged, new_actual_GMT_parity='NOT_EXECUTED',
                    producer_and_clock_receipts=receipts,
                    comparison=dict(path=str(background_path), sha256=sha256(background_path)),
                    observed_GPU_process_wall_hours=sum(c['GPU_visible_process_wall_hours'] for c in clocks.values()),
                    manual_GPU_preflight_outer_cost='NOT_CAPTURED_SEPARATE_FROM_TWO_MEASURED_PRODUCERS',
                    download_and_environment_setup='CPU_NETWORK_SETUP_NOT_TRACKER_GPU_RUNTIME',
                    population='first24 uniform TRAIN pairs,11fit scenes; failed pairs retained; no DEV5/TEST selection',
                    provenance=source)
    summary = background['summary']
    table = '\n'.join(f"|{k}|{v['valid']}/{v['pairs']}|{','.join(map(str,v['scenes_with_valid'])) or '无'}|" for k,v in summary.items())
    raw_omega_valid = sum(r['outputs']['AB']['heldout_background']['valid'] for r in runs['omega512'])
    duplicate_count = sum(p['dropped_duplicate_or_many_to_one'] for p in background['pair_audits'])
    enough_eval = sum(p['eval_count'] >= 20 for p in background['pair_audits'])
    support_note = (f'原始初步Ω512正序gate为{raw_omega_valid}/24；去除两侧重复端点/多对一后为'
                    f'{summary["omega512_AB"]["valid"]}/24，全部24个物理图像对保留，阈值未变。'
                    f'去除{duplicate_count}条重复支持后，仅{enough_eval}/24对拥有至少20个独立eval端点。'
                    '这是对应点权重/有效样本集合的工程修正，不把两套数字混为完全相同分母。'
                    '通过样本仅来自场景8，未达到预注册的至少3个时刻/2个fit场景扩展参考。'
                    '两个通过对的Ω正序P90残差仍约157/227原像素，不能把中位数gate当作全部对应正确。\n')
    common = ('416主实验：**BLOCKED_WEIGHT_ACCESS / NOT_EXECUTED**。512仅为单独的固定样本诊断，'
              '不替代416，不代表身份或跟踪成功。所有24组及失败样本均保留。\n')
    documents = {
      'OMEGA_PHASE0_INTERFACE_AUDIT.md': '# Ω Phase0 接口审计\n\n'+common+
        '\n官方源码固定48b23c8c，原始512文件实际SHA与官方LFS值一致；公开分发来源及许可记录见SOURCE_AUDIT和acquisition receipt。'
        '1411项state严格加载，全部冻结。24组×AB/BA=48次真实前向均输出有限pose、camera、depth和confidence。'
        '实际shape见omega_camera_pose_comparison.csv。外参camera-from-world，confidence=1+exp(logit)，未sigmoid或当成身份概率。\n\n'
        '独立torch2.6.0+cu118环境，V100sm70；其默认BF16检测包含模拟并返回True，native-only为False。'
        '使用官方autocast=False构造参数+外层FP16，官方fp32相机/深度head guard保留；没有修改官方源码/权重。'
        '10项坐标/合成几何/权重拒绝/AMP契约通过，真实24对两种分辨率1986点逆映射最大误差4.55e-13像素。'
        '像素中心半像素修正和连续边界矩阵分别记录，裁剪丢失/padding逆映射为invalid。\n\n'
        '新旧VGGT均在同一次forward处理物理AB或BA；GMT从未运行或改序。源码SHA、环境freeze、实际命令和完整进程计时均由原始manifest/receipt绑定。\n',
      'OMEGA_CAMERA_GEOMETRY_REPORT.md': '# Ω 相机几何诊断\n\n'+common+
        '\n以同一次联合推理的world-to-camera计算R_BA、t_BA和F；先验证SO(3)、焦距和数值基线，'
        '再用A_B.T@F_network@A_A还原原图坐标。独立forward的绝对外参不逐项比较。'
        'AB/BA的相对姿态、F尺度/符号转置差异及无标签弱框分数差异保留在原始逐对JSON。'
        '相机GT姿态误差未测；矩阵存在/数值有限不等于可靠几何。\n\n'
        '|方法/输入顺序|共同独立背景gate通过|通过场景|\n|---|---:|---|\n'+table+'\n\n'
        '旧VGGT使用真实518×518 pad接口，Ω512使用真实balanced形状；不同分辨率、有效像素支持及输出head范围限制架构归因。'
        '原任务要求的416对照尚未执行。\n\n'+support_note,
      'OMEGA_BACKGROUND_VALIDITY_REPORT.md': '# Ω 独立背景验证\n\n'+common+
        '\nSIFT与检测框屏蔽复用原geometry_confidence实现；原有all-point F/H及自拟合指标被丢弃。'
        '保存的putative matches按两侧原图坐标去重，避免同一SIFT位置多方向或many-to-one重复计数；'
        '不使用GT、模型残差或RANSAC mask选取评估点。4×4固定棋盘块分fit/eval，传统F只在fit拟合，'
        '所有模型共同在eval评价。这个工程去重规则在查看首批指标前提交，旧原始预处理指标保持不变。\n\n'
        '固定主参考：至少20个eval点、12原像素内比例≥.25、中位数≤12原像素、凸包覆盖≥完整原图面积1%。'
        '阈值未降低；归一化残差单列。匹配不足、覆盖不足和预测F不一致分开记录，不能从这些数字断言无物理重叠。'
        '描述子匹配是可复核的putative对应，并非真实相机或三维GT。\n\n'
        '|方法/顺序|通过/全部|通过场景|\n|---|---:|---|\n'+table+'\n\n'
        +support_note+'\n详见omega_background_geometry.csv和omega_distinct_heldout_comparison_v1.json，全部失败对保留。'
        '没有通过集时不计算虚假100%准确率。512单独诊断不准入416主实验后续阶段。\n',
      'OMEGA_IDENTITY_CORRECTION_REPORT.md': '# Ω 身份纠错：NOT_EXECUTED\n\n'+common+
        '\n没有执行几何候选身份分离、logits修正、联合求解/Bank/最终ID干预；纠错和破坏均NOT_MEASURED，不能填0。'
        '没有深度置信度残差校准，没有深度身份收益。G0–G5为明确条件未执行行。\n\n'
        '复用原始169468个主查询/1553个已认证主关联wrong/missed：650有同步对侧原生参考、314有同步纯净正例，'
        '297的同步正例在Top8。903没有同步对侧原生参考，当前F不可直接适用。'
        '缺少同步纯正例限制本次认证正/负候选比较，不等于所有未知历史都是真错误或一个普遍纠错上界。'
        'fit11为1527错误/304同步纯正例；DEV12为26/10，23个全历史纯正例不能当成23个同步正例。'
        '这些是已有原生输入的适用性统计，绝非Ω修复数量。\n\n'
        '没有重跑全TRAIN、重新匹配GT或补理想历史。DEV12标签已接触，26错误高度相关，不能称盲测；'
        '后续必须先fit固定方案再诊断DEV12。\n',
      'OMEGA_ONLINE_TRACKING_REPORT.md': '# Ω 在线跟踪：NOT_EXECUTED\n\n'+common+
        '\n前序主几何/身份gate尚未满足，未接入GMT get_asso，未运行完整视频或DEV5/TEST。'
        'B0–B6当前新实验为NOT_EXECUTED，HOTA/AssA/CVIDF1/IDSW/DetA/MOTA/CVMA/FPS均NOT_MEASURED。'
        '既有基线仍保留在历史报告中，不冒充本轮Ω结果。\n\n'
        '原GTRRCNN、config、GeometryRuntime、pair_score_adapter字节与c6999a7a完全一致。'
        '这证明源码范围，不能替代一个没有执行的新实际GMT输出parity实验。'
        'Ω Adapter目前仅供独立诊断，默认跟踪路径从未调用它。没有AG/QD/JEV/蒸馏、主干训练、GT修复或跨时间F传播。\n',
      'OMEGA_FINAL_RESEARCH_REPORT.md': '# VGGT-Ω 当前研究交付：主实验尚未完成\n\n'+common+
        '\n状态 **DIAGNOSTICS_COMPLETE_MAIN416_ACCESS_BLOCKED**。不是模型全面无效的结论，'
        '也不是OMEGA_GEOMETRY/IDENTITY/TRACKING/EFFICIENCY_SUCCESS。已完成隔离分支、官方源码和许可审计、'
        '精确预处理/合成几何/真实检测框坐标验证、原生同步适用性统计与96组预先冻结索引；'
        '已获取并核验公开512原版字节，完成24组48次双向真实Ω调用及同物理图像的旧VGGT/传统背景对照。\n\n'
        '|方法/顺序|独立背景gate通过/全部|通过场景|\n|---|---:|---|\n'+table+'\n\n'
        +support_note+'\n通过背景残差不证明身份区分；没有执行身份打分/求解干预、完整视频或新增GMT速度实验。'
        '其余72组扩展、深度身份用途和全部在线分支均明确NOT_EXECUTED。'
        '首24组覆盖11个fit场景而非独立人物样本，原GMT见过DEV5，DEV12标签已接触。'
        '未知/失败样本保留，阈值和原GT未修改。\n\n'
        '实际两生产进程端到端秒、P50/P95预处理+forward+检查、CUDA分配器/保留峰值见omega_runtime.csv。'
        '这些时间包括重复AB/BA的有限几何审计，没有GMT，不是系统FPS；旧VGGT额外输出world_points，不能将总成本差直接归为架构。'
        '被测两生产进程以外的手动CUDA探针外层成本未捕获，环境/下载另列，不伪造完整历史总账。\n\n'
        '继续主实验需要合法取得官方416 reproduction原始权重，SHA '
        '`e900833330eb0e570e10543e6108d0dd704fa7c8dbc32738a645987875298896`。'
        '解锁后复用同一冻结cohort，先运行416首24组；只有足够可靠几何及净真实身份决策收益后才扩大或接入GMT。'
        '不能用512或GGUF转换结果替换这项待办。\n'
    }
    write_csv(ROOT/'results/omega_camera_pose_comparison.csv', pose_rows)
    write_csv(ROOT/'results/omega_runtime.csv', runtime)
    write_csv(ROOT/'results/omega_identity_candidate_scores.csv', condition_rows)
    write_csv(ROOT/'results/omega_full_tracking_metrics.csv', tracking)
    save_new(ROOT/'results/omega_identity_correction_audit.json', identity)
    for name, content in documents.items():
        with (ROOT/'docs'/name).open('x') as f:
            f.write(content)
    manifest['published_artifacts'] = {name:sha256(ROOT/name) for name in targets if name != 'results/omega_experiment_manifest.json'}
    save_new(ROOT/'results/omega_experiment_manifest.json', manifest)
    print(json.dumps(dict(status=manifest['status'], scientific_goal_complete=False,
                         measured_producer_GPU_wall_hours=manifest['observed_GPU_process_wall_hours'])))


if __name__ == '__main__':
    main()

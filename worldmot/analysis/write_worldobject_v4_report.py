#!/usr/bin/env python3
"""Publish measured V4 artifacts or explicit pending states, never fake scores."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import shutil
import xml.etree.ElementTree as ET
from worldobject_v4_audit_common import code_fingerprint,write_json


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--cpu-junit',type=Path,required=True);p.add_argument('--measurements',type=Path)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2];out=repo/'worldmot/results';docs=repo/'worldmot/docs'
    fp=code_fingerprint(repo);junit=ET.parse(a.cpu_junit).getroot();suites=list(junit.iter('testsuite'))
    tests=sum(int(s.attrib.get('tests',0)) for s in suites);failures=sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)
    skipped=sum(int(s.attrib.get('skipped',0)) for s in suites)
    from worldmot.data.worldobject_cache import sha256
    receipt=read(a.runtime/'cpu_receipt.json')
    if (receipt['status']!='PASSED' or receipt['source_sha256']!=fp
            or receipt['junit_sha256']!=sha256(a.cpu_junit)):
        raise ValueError('CURRENT_SOURCE_BOUND_CPU_RECEIPT_REQUIRED')
    cpu={'status':'PASSED' if not failures else 'FAILED','tests':tests,'failures_and_errors':failures,'skipped':skipped,
        'time_s':sum(float(s.attrib.get('time',0)) for s in suites),'junit_sha256':sha256(a.cpu_junit),
        'source_sha256':fp,'python':platform.python_version(),'scope':'CPU synthetic and source-bound compatibility tests; no real identity scores',
        'legacy_compatibility':'80 synthetic timestamps exactly match frozen b448af4 head inputs, decisions and numeric state; real checkpoint replay pending',
        'replay_integration':'all15 registered paths, 12 factorial edges and fixed-history G3 invariance exercised with explicitly synthetic unit-test weights'}
    write_json(out/'v4_cpu_test_report.json',cpu)
    names=['v4_gmt_synchronous_identity_comparison.json','v4_crossview_hard_cases.json','v4_frozen_geometry_factorial.json',
        'v4_closed_loop_identity_attribution.json','v4_error_correction_attribution.json','v4_runtime_analysis.json']
    pending={'status':'WAITING_FOR_NAS_POWER_RESTORATION','reason':'user reports NAS power outage; retry no earlier than 2026-10-26 04:00 Asia/Shanghai',
        'no_metric_inferred':True,'optimizer_started':False,'V4_formal_TEST_started':False}
    for name in names:
        path=out/name;measured=a.measurements/name if a.measurements else None
        if measured and measured.is_file():shutil.copyfile(measured,path)
        elif not path.exists() or read(path).get('status','').startswith('WAITING'):write_json(path,pending)
    comparison=out/'v4_train_val_comparison.csv';measured=a.measurements/'v4_train_val_comparison.csv' if a.measurements else None
    if measured and measured.is_file():shutil.copyfile(measured,comparison)
    elif not comparison.exists():
        from worldmot.tracking.world_object_v4_variants import state_configs,factorial_configs
        with comparison.open('w',newline='') as f:
            writer=csv.DictWriter(f,['variant','HOTA','AssA','IDF1','status','scope'],lineterminator='\n');writer.writeheader()
            for variant in [*state_configs(),*(k for k in factorial_configs() if k!='G1_1_G2_1_G3_1'),'R2_NO_GEOMETRY_REFERENCE']:
                writer.writerow({'variant':variant,'status':'NOT_MEASURED_NAS_OFFLINE','scope':'TRAIN only; no scores yet'})
    A=read(out/names[0]);B=read(out/names[2]);D=read(out/names[3]);R=read(out/'v4_runtime_analysis.json')
    gate={'schema':'worldmot_v4_stage_gate_v1','generated_utc':datetime.now(timezone.utc).isoformat(),
        'branch':'research/worldmot-v4-persistent-state-20261025','base_commit':'b448af4a7dfdf738aeab0c9dd5bc01e9a875beac',
        'source_sha256':fp,'stages':{'A_GMT_SYNCHRONOUS_IDENTITY_AUDIT':A['status'],
        'B_FROZEN_GEOMETRY_CAUSAL_AUDIT':B['status'],'C_WORLD_OBJECT_STATE_IMPLEMENTATION':'CPU_PASSED_REAL_VERIFICATION_PENDING' if not failures else 'CPU_FAILED',
        'D_TRAIN_VAL_STATE_VERIFICATION':D['status']},'CPU':cpu,'scientific_gain_proven':False,
        'full_training_qualified':False,'new_full_training_authorized':False,'new_V4_formal_TEST_authorized':False,
        'R2_R3_predictions_or_weights_modified':False,'current_limit':'no new real V4 measurement until immutable NAS inputs can be verified'}
    if D.get('status')=='COMPLETE':
        variants=D['variants'];base=variants['C0']['original_legal_TRAIN_VAL_TrackEval'];candidate=variants['C4']['original_legal_TRAIN_VAL_TrackEval']
        diag=variants['C4']['paired_against_C0']
        eligible=[]
        for key in ('C1','C2','C3','C4'):
            m=variants[key]['original_legal_TRAIN_VAL_TrackEval'];d=variants[key]['paired_against_C0']
            if (not failures and all(m[k]>base[k] for k in ('HOTA','AssA','IDF1'))
                and d.get('temporal_corrections',0)>d.get('temporal_regressions',0)
                and d.get('same_person_corrections',0)>=d.get('same_person_regressions',0)
                and d.get('different_person_corrections',0)>=d.get('different_person_regressions',0)):eligible.append(key)
        chosen=max(eligible,key=lambda k:(variants[k]['original_legal_TRAIN_VAL_TrackEval']['AssA'],
            variants[k]['original_legal_TRAIN_VAL_TrackEval']['IDF1'],variants[k]['original_legal_TRAIN_VAL_TrackEval']['HOTA'],-int(k[1]))) if eligible else None
        qualified=bool(eligible) and B.get('status')=='COMPLETE' and A.get('status')=='COMPLETE'
        gate['stages']['C_WORLD_OBJECT_STATE_IMPLEMENTATION']='CPU_AND_REAL_C0_COMPATIBILITY_PASSED'
        gate.update(scientific_gain_proven=bool(eligible),full_training_qualified=qualified,
            TRAIN_eligible_state_variants=eligible,TRAIN_selected_state_variant=chosen,
            C5_live_quality_gate_passed='C4' in eligible,
            current_limit='TRAIN internal evidence only, final head saw all24; separate user authorization required for full next phase')
    write_json(out/'v4_stage_gate.json',gate)
    lines=['# WorldMOT V4 — master research report','',
        '本报告严格区分已通过的工程测试与尚未完成的真实实验。R2/R3 的原结果、权重与感知缓存不变。','',
        f'当前 CPU 测试：**{tests} 项，失败 {failures} 项，跳过 {skipped} 项**。',
        f'执行代码指纹：`{fp}`。','',
        'GMT 原始完整轨迹已从原正式评测证据定位到 `fulltrain_20261022/test/gmt_native`，仍需恢复后逐文件复核 Manifest/轨迹 SHA。检测候选缓存不会作为 GMT Native Global ID。','',
        '| 阶段 | 实际状态 |','|---|---|']
    lines += [f'| {k} | {v} |' for k,v in gate['stages'].items()]
    lines += ['','实现包括独立 G1/G2/G3/G4 控制、六维常速度预测、Joseph Kalman 更新、身份/空间分离、置信度门控长短期视觉记忆、保守多视角测量和因果 Ω 请求。',
        'C0 默认路径用冻结 R2 源码对照；CPU 合成测试的 80 时刻完全一致不等价于真实 checkpoint 的全视频复现。',
        '完整 TRAIN 回放必须再次验证 C0 的每个输入张量、决策与数值状态。','',
        '冻结身份头的最终 all24 TRAIN 权重已经看过六个 TRAIN VAL 场景。本轮是 TRAIN 内部机制开发，不是独立泛化验证。',
        '正式 TRAIN VAL TrackEval 仅使用原始 GT 合法的 4/8/12；16/20/24 仅作干净标注子集诊断。',
        '默认时间单位为 frame_id tick，空间单位为相对世界坐标；不假定秒或米。','',
        '未获得真实指标时，不能回答 V4 是否超过 R2、No-Geometry 或 GMT。CSV 留空未测值，不填零。',
        '缓存请求次数不等于真实 FPS；稀疏请求后的 Live keyframe 对齐和 RGB-to-ID 延迟必须另测。','',
        '下一步：NAS 恢复后核验空间与不可变输入，先运行有限真实工程预检，随后完成预注册六场闭环及单步归因。',
        '本轮不启动新的 24 TRAIN 长训，不执行 V4 正式 TEST 跟踪或 TEST 参数选择。','',
        '来源与许可详见 [V4_SOURCE_ADAPTATION.md](V4_SOURCE_ADAPTATION.md)；具体固定范围见 [V4_PREREGISTRATION.md](V4_PREREGISTRATION.md)。']
    if D.get('status')=='COMPLETE':
        lines += ['','## 已完成 TRAIN 内部指标','','| 方法 | HOTA | AssA | IDF1 |','|---|---:|---:|---:|']
        for k,v in D['variants'].items():
            m=v['original_legal_TRAIN_VAL_TrackEval'];lines.append(f"| {k} | {m['HOTA']:.4f} | {m['AssA']:.4f} | {m['IDF1']:.4f} |")
        lines += ['',f"预注册 TRAIN 验收门槛：{'通过' if gate['full_training_qualified'] else '未通过'}；候选：{gate.get('TRAIN_eligible_state_variants',[])}。所有负结果及新增回归均保留。"]
    (docs/'V4_MASTER_RESEARCH_REPORT.md').write_text('\n'.join(lines)+'\n')
    stage_docs={'V4_STAGE_A_SYNC_IDENTITY.md':('阶段 A：公平同步身份核验',A['status'],
        'Protocol A 使用各自成功匹配的事件；Protocol B 使用与 GMT 共同 GT 事件。报告 TP/FP/FN、Precision、Recall、F1 和歧义掩码，不用总体 Accuracy 掩盖类不均衡。'),
        'V4_STAGE_B_GEOMETRY_CAUSAL_AUDIT.md':('阶段 B：冻结几何因果归因',B['status'],
        'B1 从同一 C0 历史复制、丢弃反事实状态；B2 从场景起点独立闭环。G3 单步不应改变当前分配。G4 置零仅测固定头的输入敏感性，不能据此证明独立学习收益。'),
        'V4_STAGE_C_PERSISTENT_STATE.md':('阶段 C：持续世界对象实现',gate['stages']['C_WORLD_OBJECT_STATE_IMPLEMENTATION'],
        '先预测世界状态，再联合分配；身份与空间分别更新。空间冲突不会仅因过期中心硬拒绝可靠身份。C4 同一 Ω 调用的两路观测只合成一次保守更新。C5 请求只看当前视觉和过去状态。')}
    for name,(title,status,description) in stage_docs.items():
        (docs/name).write_text(f'# {title}\n\n实际状态：`{status}`。\n\n{description}\n\n真实测量、纠正与回归均以 results 中的完成报告为准。未测指标不使用历史 R2 数值代填。\n')
    write_json(a.runtime/'last_report_state.json',gate)
    print(json.dumps({'CPU_tests':tests,'CPU_failures':failures,'stages':gate['stages']}))


if __name__=='__main__':main()

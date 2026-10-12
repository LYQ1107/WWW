"""Read-only history/weight/cache registration; historical scores stay diagnostic."""
import collections
from jev_oneweek_common import *


def main():
    protect()
    entries = []
    docs = ['JEV_PHASE6_FINAL_REPORT.md', 'PHASE7_FINAL_RESEARCH_REPORT.md',
            'PHASE10_FINAL_RESEARCH_REPORT.md', 'JEV_PHASE12_FINAL_RESEARCH_REPORT.md',
            'JEV_PHASE13_FINAL_RESEARCH_REPORT.md', 'JEV_PHASE14_FINAL_RESEARCH_REPORT.md',
            'JEV_PHASE15_FINAL_RESEARCH_REPORT.md', 'PHASE17_FINAL_RESEARCH_REPORT.md',
            'PHASE19_FINAL_RESEARCH_REPORT.md', 'PHASE20_FINAL_RESEARCH_REPORT.md',
            'WWW_JEV_MASTER_REQUIREMENTS_AUDIT.md']
    for name in docs:
        path = ROOT/'docs'/name
        if path.exists():
            text = path.read_text()
            entries.append(dict(kind='historical_report', asset=ref(path),
                classification='REUSE_DIAGNOSTIC', read_lines=len(text.splitlines()),
                reason='Different historical actor/frontend/training budgets; not a current unified main-table row'))
        else:
            entries.append(dict(kind='historical_report', path=str(path), classification='NOT_AVAILABLE'))
    # Every phase has a summary-index inventory, without reading large GT/trace assets.
    summaries = []
    for phase in range(6,22):
        folder = ROOT/'reports'/f'JEV_PHASE{phase}'
        files = [ref(p) for p in folder.glob('*.json')]
        summaries.append(dict(phase=phase, summary_files=len(files), manifest=files,
                              classification='REUSE_DIAGNOSTIC' if files else 'NOT_AVAILABLE'))
    weights = []
    def checkpoint(label, descriptor, use):
        path = Path(descriptor['path'])
        if not path.exists():
            weights.append(dict(label=label, expected=descriptor, classification='NOT_AVAILABLE'))
            return
        actual = ref(path)
        assert actual['SHA256'] == descriptor['SHA256'], label
        weights.append(dict(label=label, asset=actual, classification=use, verified=True))
    from jev_phase13_common import STAGE1, STAGE1_SHA, STAGE2, STAGE2_SHA
    checkpoint('GMT Stage1 perception', dict(path=str(STAGE1), SHA256=STAGE1_SHA), 'REUSE_VALID')
    checkpoint('Original complete GMT Stage2', dict(path=str(STAGE2), SHA256=STAGE2_SHA), 'REUSE_VALID')
    for variant in ['fixed_question','multi_question','set_transformer']:
        for seed in SEEDS:
            path = PHASE14/'formal_v2'/variant/'availability_joint'/f'seed{seed}'/'RESULT.json'
            result = read(path)
            assert result['status']=='COMPLETE' and result['actual_updates']==20000
            checkpoint(f'XIV {variant} seed{seed}', result['checkpoint'], 'REUSE_VALID')
    source = read(ROOT/'reports/JEV_PHASE13/SUPERVISED_TRAINING.json')
    def visit(value):
        if isinstance(value,dict):
            if 'checkpoint' in value and value.get('variant') in ('full','fixed_question','set_transformer'):
                checkpoint(f'XIII {value["variant"]} seed{value.get("seed")}', value['checkpoint'], 'REUSE_VALID')
            for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    visit(source)
    path = PHASE15/'training_full_payload_v3/F_full/seed20261009/pilot/RESULT.json'
    if path.exists():
        result=read(path)
        for key in ('checkpoint','last_checkpoint'):
            if key in result:
                checkpoint('XV V3 F_full seed20261009 pilot',result[key],'REUSE_DIAGNOSTIC')
                break
    datasets=[]
    for video in TRAIN:
        path = PHASE15/'commitment_dataset_v2'/f'video{video:02d}'/'RESULT.json'
        result=read(path); assert result['status']=='PASS'
        descriptor=result['DATASET']; assert sha(descriptor['path'])==descriptor['SHA256']
        datasets.append(dict(video=video,source=ref(path),tensor=descriptor,
                             records=result['records'],classification='REUSE_VALID',
                             purpose='Reuse causal current/past tensors and offline masks; new per-view labels must be audited'))
    caches=[]
    for video in TRAIN+DEV:
        path=PHASE13/'stage1_cache_v1'/f'video{video:02d}'
        result=read(path/'RESULT.json')
        index=ref(path/'index.jsonl')
        assert index['SHA256']==result['cache_index_SHA256']
        caches.append(dict(video=video,index=index,results=ref(path/'RESULT.json'),
                           classification='REUSE_VALID',role='Common VFCE1024 perception; frozen per-payload SHA verified on load'))
    oldcache=Path('/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train')
    coverage=collections.Counter(); dimensions=collections.defaultdict(set)
    with (oldcache/'index.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['video_id'] in TRAIN+DEV:
                coverage[row['video_id']]+=1;dimensions[row['video_id']].add(row['reid_dim'])
    original=dict(index=ref(oldcache/'index.jsonl'),allowed_video_payloads=dict(coverage),
                  allowed_video_dimensions={str(k):sorted(v) for k,v in dimensions.items()},
                  distinction='Original full Stage2 perception cache with VFCE+RPCE1152; not common Stage1 VFCE1024',
                  restoration_status='RERUN_REQUIRED_NATIVE_PARITY')
    registry=dict(status='COMPLETE', binding=binding('Read-only historical asset and cache audit'),
                  entries=entries, phase_summary_inventory=summaries, weights=weights,
                  datasets=datasets, common_caches=caches, original_GMT_cache=original,
                  rejected_methods=['XVI hard quarantine','XVII unsafe soft-memory','learned MEMORY WRITE without real reward'],
                  original_MASTER_safety_gate_unchanged=True,
                  fair_MLP_20k='NOT_AVAILABLE; current equal-budget ordinary residual controls must be trained',
                  score_reuse='All historical metrics diagnostic until exact current conditions and raw prediction parity are proved',
                  old_results_overwritten=False)
    save(REPORTS/'REUSED_EVIDENCE_REGISTRY.json',registry)
    facts=[('VII fixed rule',85.0300,'B2 84.9749; original heldouts09/10/11, not current main table'),
           ('XIII Fixed Question',75.637,'Full75.056; different learned WHO protocol'),
           ('XIV Multi',73.505,'AssA73.639/CVIDF1 88.546/IDSW391.667; seed20261009 IDSW824 retained'),
           ('XV V3',74.188,'AssA74.891/CVIDF1 90.286/IDSW147; pilot diagnostic'),
           ('XVI hard memory',61.764,'IDSW1494; rejected'),
           ('Original full GMT',78.110,'Separate trained detector/VFCE/RPCE path; never called same frontend')]
    report='# 历史资产审计\n\n状态：完成只读资产核验；尚无本周性能结论。\n\n'
    report+=f'已核验 {len(weights)} 个必要历史权重、4个真实TRAIN张量数据集及7个共同Stage1缓存索引。无20k重训、无大型数据下载。完整路径、SHA、缺失与复用类别见 REUSED_EVIDENCE_REGISTRY.json。\n\n'
    report+='| 历史方法 | HOTA | 边界 |\n| --- | ---: | --- |\n'
    for name,value,note in facts:report+=f'| {name} | {value} | {note} |\n'
    report+='\n上述不同阶段指标不可拼成同条件主表。MLP没有对应20k正式权重，不能用历史弱/失败MLP替代公平新控制。原始GMT的1152D旧TRAIN缓存仍在，可尝试真实原生恢复；它与1024D Stage1共同缓存的差异必须单列。\n'
    (DOCS/'HISTORICAL_EVIDENCE_AUDIT.md').write_text(report)
    jobs=read(REPORTS/'EXPERIMENT_REGISTRY.json');jobs['jobs']['historical_asset_audit'].update(status='COMPLETE',results=ref(REPORTS/'REUSED_EVIDENCE_REGISTRY.json'));save(REPORTS/'EXPERIMENT_REGISTRY.json',jobs)
    print('ONEWEEK_ASSET_AUDIT_COMPLETE',len(weights),len(datasets),len(caches),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        failure('asset_audit',traceback.format_exc());raise

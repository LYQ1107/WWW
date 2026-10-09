"""Hash every historical model artifact referenced by saved research reports."""
import collections
import time
from jev_phase14_common import *

def collect(value, found, evidence, key=''):
    if isinstance(value, dict):
        p = value.get('path')
        s = value.get('SHA256', value.get('sha256'))
        if isinstance(p, str) and p.endswith(('.pth', '.pt', '.jit')) and any(
                w in (key + p).lower() for w in ['checkpoint', 'model', 'last', 'best', 'training', 'policy']):
            if not any(w in p.lower() for w in ['dataset', 'prefix', 'snapshot', 'recovery_slot']):
                found[p].add(s) if s else found[p]
                evidence[p].add(key)
        for k, v in value.items():
            collect(v, found, evidence, key + '/' + str(k))
    elif isinstance(value, list):
        for v in value:
            collect(v, found, evidence, key)

def main():
    protect()
    found = collections.defaultdict(set)
    evidence = collections.defaultdict(set)
    for p in sorted((ROOT / 'reports').glob('JEV_PHASE*/*.json')):
        if '/JEV_PHASE14/' not in str(p):
            collect(read(p), found, evidence, str(p.relative_to(ROOT)))
    # Include intermediate LAST/BEST and superseded Phase XIII model artifacts.
    for version in ['20261009_v1', '20261009_v2']:
        root = OLD.parent / version
        for p in root.glob('*training*/*/seed*/*.pth'):
            found[str(p)]
    from jev_phase13_common import STAGE1, STAGE2, STAGE1_SHA, STAGE2_SHA
    found[str(STAGE1)].add(STAGE1_SHA)
    found[str(STAGE2)].add(STAGE2_SHA)
    result = []
    begin = time.monotonic()
    for i, (name, expected) in enumerate(sorted(found.items()), 1):
        p = Path(name)
        item = {'path': name, 'recorded_SHA256': sorted(expected),
                'source_report_references': sorted(evidence[name])}
        if p.is_file():
            item.update(actual_SHA256=sha(p), bytes=p.stat().st_size,
                        status='PASS' if not expected or sha(p) in expected else 'HASH_MISMATCH')
        else:
            item.update(status='MISSING_HISTORICAL_ARTIFACT', actual_SHA256=None)
        result.append(item)
        if i % 20 == 0:
            save(OUT / 'checkpoint_audit/PROGRESS.json', {'status': 'RUNNING', 'done': i, 'total': len(found)})
            print('CHECKPOINT_HASH', i, len(found), flush=True)
    counts = collections.Counter(r['status'] for r in result)
    save(REPORTS / 'HISTORICAL_CHECKPOINT_AUDIT.json', dict(
        status='FAIL' if counts['HASH_MISMATCH'] else 'COMPLETE_WITH_MISSING_HISTORY' if counts['MISSING_HISTORICAL_ARTIFACT'] else 'PASS',
        binding=binding(), inventory_scope='all model files referenced by top-level Phase V-XIII reports plus all extant Phase XIII intermediate training snapshots; missing historical paths explicitly retained',
        counts=dict(counts), models=result, seconds=time.monotonic()-begin))
    print('CHECKPOINT_AUDIT_COMPLETE', dict(counts), flush=True)

if __name__ == '__main__':
    main()

"""Post-screen age stratification with saved heads, no training or selection.

The historical audit used max_age20, whereas the new preregistration uses200.
This explains scope differences without changing the new primary screen gate.
"""
import argparse
import json
from pathlib import Path

import torch

from quick_cv_pil_native import load_native, native_episodes, report
from quick_cv_pil_probe import digest, LowRankPredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--first-result', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(1)
    first = json.loads(a.first_result.read_text())
    audit = json.loads(a.audit.read_text())
    assert digest(a.first_result) == audit['original_result_sha256']
    data = load_native(a.runtime, 12)
    episodes = native_episodes(data)
    assert len(episodes) == first['counts']['12']
    models = {}
    for arm, checkpoint in audit['local_checkpoints'].items():
        assert digest(checkpoint['path']) == checkpoint['sha256']
        saved = torch.load(checkpoint['path'], weights_only=True, map_location='cpu')
        assert saved['steps'] == 300 and saved['seed'] == 20261015
        m = LowRankPredictor(nonlinear=arm != 'linear', use_camera=arm != 'predictive_no_camera')
        m.load_state_dict(saved['state_dict'], strict=True)
        models[arm] = m.eval()
    subsets = dict(all_age_le200=episodes,
        age_le20=[e for e in episodes if e['frame']-e['source_frame'] <= 20],
        age_gt20=[e for e in episodes if e['frame']-e['source_frame'] > 20])
    results = {}
    for name, ep in subsets.items():
        rows = dict(native=report(data, ep, torch.device('cpu'), mode='native'),
                    native_plus_GMT_cosine=report(data, ep, torch.device('cpu')))
        rows.update({arm: report(data, ep, torch.device('cpu'), model=m) for arm, m in models.items()})
        base = rows['native']['correct']
        for arm, r in rows.items():
            r['fixes_vs_native'] = sum(not b and x for b, x in zip(base, r['correct']))
            r['harms_vs_native'] = sum(b and not x for b, x in zip(base, r['correct']))
        if name == 'all_age_le200':
            for arm, r in rows.items():
                expected = first[arm] if arm in ('native', 'native_plus_GMT_cosine') else first['methods'][arm]
                assert all(r[k] == expected[k] for k in ('recall1', 'mrr', 'clauses', 'unique_queries', 'persons'))
        results[name] = {arm: {k: v for k, v in r.items() if k != 'correct'} for arm, r in rows.items()}
    result = dict(status='PASS_SAVED_HEAD_METRIC_PARITY_NO_RETRAINING',
        audit_sha256=digest(a.audit), source_sha256=digest(__file__),
        results=results, primary_gate_unchanged=first['gate'],
        scope='Post hoc descriptive strata; no checkpoint selection and no changed primary protocol.',
        limitations='New max_age20 stratum still applies new eligibility/deduplication; historical clauses are a different estimand. No online improvement claim.')
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

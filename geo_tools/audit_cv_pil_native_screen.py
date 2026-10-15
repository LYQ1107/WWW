"""Reproduce the locked CPU screen and retain paired attribution, without tuning.

The first native_screen.json stays immutable. Replays the same seed/300 steps,
requires exact aggregate equality, and saves small heads locally for review.
No encoders, videos, online tracking or extra hyperparameters are introduced.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

import quick_cv_pil_native as native
from quick_cv_pil_probe import SEED, digest, LowRankPredictor


def paired(episodes, before, after):
    fixes = [i for i, (a, b) in enumerate(zip(before, after)) if not a and b]
    harms = [i for i, (a, b) in enumerate(zip(before, after)) if a and not b]
    changed = fixes + harms
    return dict(fixes=len(fixes), harms=len(harms),
                affected_unique_queries=len({episodes[i]['query'] for i in changed}),
                affected_persons=len({episodes[i]['target_id'] for i in changed}),
                fix_clause_indices=fixes, harm_clause_indices=harms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--first-result', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    first_sha = digest(a.first_result)
    first = json.loads(a.first_result.read_text())
    started = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    data = {v: native.load_native(a.runtime, v) for v in (5, 8, 12)}
    episodes = {v: native.native_episodes(data[v]) for v in data}
    coverage = {}
    for v, d in data.items():
        ep = episodes[v]
        for e in ep:
            assert e['frame'] == d['meta'][e['query']]['frame']
            assert len(e['candidate_rows']) == len(e['native_logits'])
            labels = []
            for rows in e['candidate_rows']:
                assert len(rows) == 1
                r = rows[0]
                assert 0 <= r < len(d['meta'])
                assert 0 < e['frame'] - d['meta'][r]['frame'] <= 200
                assert e['camera'] != d['meta'][r]['camera']
                labels.append(int(d['labels'][r]))
            assert sum(x == e['target_id'] for x in labels) == 1
            assert labels[e['target']] == e['target_id']
        coverage[str(v)] = dict(observations=len(d['meta']), clauses=len(ep),
            events=len(d['events']), unique_queries=len({e['query'] for e in ep}),
            video_person_groups=len({e['target_id'] for e in ep}),
            candidate_count_histogram=dict(Counter(len(e['candidate_rows']) for e in ep)),
            per_person_clause_counts=dict(Counter(e['target_id'] for e in ep)),
            source_frame_min=min(e['source_frame'] for e in ep),
            query_frame_max=max(e['frame'] for e in ep),
            strictly_past_opposite_camera=True)
    a.out.mkdir(parents=True, exist_ok=False)
    capture, checkpoints = {}, {}
    original_report, original_fit = native.report, native.fit_model
    model_arms = {}

    def fit(arm, *args, **kwargs):
        model, logs = original_fit(arm, *args, **kwargs)
        model_arms[id(model)] = arm
        checkpoint = a.out / (arm + '.pth')
        torch.save(dict(arm=arm, seed=SEED, steps=300,
                        state_dict=model.state_dict()), checkpoint)
        checkpoints[arm] = dict(path=str(checkpoint), sha256=digest(checkpoint),
                                bytes=checkpoint.stat().st_size)
        return model, logs

    def report(d, ep, device, model=None, kind='gmt', alpha=2., mode='joint'):
        result = original_report(d, ep, device, model, kind, alpha, mode)
        if d['video'] == 12:
            key = ('native' if mode == 'native' else model_arms[id(model)]
                   if model is not None else 'native_plus_GMT_cosine')
            predictions, score_rows = [], []
            with torch.no_grad():
                for e in ep:
                    scores = (torch.from_numpy(e['native_logits']).to(device)
                              if mode == 'native' else
                              native.scores(d, e, device, model, kind, alpha))
                    predictions.append(int(torch.argsort(scores, descending=True, stable=True)[0]))
                    score_rows.append(scores.cpu().tolist())
            assert result['correct'] == [i == e['target'] for i, e in zip(predictions, ep)]
            capture[key] = dict(correct=result['correct'], predictions=predictions, scores=score_rows)
        return result

    # Execute the unmodified main with observers only; no altered loss or sampling.
    native.fit_model, native.report = fit, report
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(native.__file__)), '--runtime', str(a.runtime),
                    '--out', str(a.out / 'replay'), '--steps', '300', '--device', 'cpu']
        native.main()
    finally:
        native.fit_model, native.report = original_fit, original_report
        sys.argv = old_argv
    replay_path = a.out / 'replay/native_screen.json'
    assert first == json.loads(replay_path.read_text()), 'Aggregate replay mismatch'
    assert first_sha == digest(a.first_result), 'Original result changed'
    ep = episodes[12]
    summaries = {}
    for arm, values in capture.items():
        per_query = defaultdict(list)
        for e, good in zip(ep, values['correct']):
            per_query[e['query']].append(int(good))
        summaries[arm] = dict(correct_clauses=sum(values['correct']),
            error_clauses=len(ep)-sum(values['correct']),
            error_unique_queries=sum(not all(rows) for rows in per_query.values()),
            query_macro_recall1=float(np.mean([np.mean(rows) for rows in per_query.values()])))
    comparisons = {}
    for arm in capture:
        if arm != 'native':
            comparisons[arm + '_vs_native'] = paired(ep, capture['native']['correct'], capture[arm]['correct'])
    for ref in ('contrastive', 'predictive_no_camera', 'teacher_distill'):
        if ref in capture:
            comparisons['predictive_vs_' + ref] = paired(ep, capture[ref]['correct'], capture['predictive']['correct'])
    changed = [i for i in range(len(ep)) if any(
        c['predictions'][i] != capture['native']['predictions'][i] for c in capture.values())]
    changed_rows = []
    for i in changed:
        e = ep[i]
        changed_rows.append(dict(clause_index=i, query=e['query'], frame=e['frame'],
            camera=e['camera'], person=e['target_id'], source_frame=e['source_frame'],
            source_camera=e['source_camera'], target_index=e['target'],
            reference_observations=[r[0] for r in e['candidate_rows']],
            reference_gt_labels=[int(data[12]['labels'][r[0]]) for r in e['candidate_rows']],
            scores={arm: c['scores'][i] for arm, c in capture.items()},
            predicted_indices={arm: c['predictions'][i] for arm, c in capture.items()}))
    teacher_gate = {}
    for v in (5, 8):
        raw = original_report(data[v], episodes[v], torch.device('cpu'), mode='native')['correct']
        fused = original_report(data[v], episodes[v], torch.device('cpu'), kind='dinov2')['correct']
        teacher_gate[str(v)] = dict(native_correct=sum(raw),
            native_plus_dinov2_cosine_correct=sum(fused),
            paired=paired(episodes[v], raw, fused))
    fit_ep = episodes[5] + episodes[8]
    indices = np.random.default_rng(SEED).integers(len(fit_ep), size=300).tolist()
    fit_groups = {(e['video'], e['target_id']) for e in fit_ep}
    root = Path(__file__).resolve().parents[1]
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    sources = {name: digest(Path(__file__).with_name(name)) for name in (
        'quick_cv_pil_native.py', 'quick_cv_pil_probe.py', 'audit_cv_pil_native_screen.py')}
    audit = dict(status='PASS_EXACT_FIXED_CONFIGURATION_REPLAY', source_commit=source_commit,
        original_result_sha256=first_sha, replay_result_sha256=digest(replay_path),
        source_sha256=sources, seed=SEED, steps_per_arm=300, device='cpu',
        torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
        started_utc=start_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
        replay_and_audit_wall_seconds=time.monotonic()-started, gpu_hours=0,
        parameters_per_head=sum(p.numel() for p in LowRankPredictor().parameters()),
        coverage=coverage, fit_video_person_groups=len(fit_groups),
        shared_episode_schedule_sha256=__import__('hashlib').sha256(json.dumps(indices).encode()).hexdigest(),
        teacher_gate_training_only=teacher_gate, methods=summaries,
        paired_comparisons=comparisons, changed_clauses=changed_rows,
        local_checkpoints=checkpoints,
        raw_gate=first['gate'], decision='REVIEW_ONLY_NO_AUTOMATIC_FULL_VIDEO_RUN',
        limitations=[
            'One fixed seed repeated deterministically; this is not an independent trial.',
            'Video12 is previously exposed; repeated clauses are dependent; only six people.',
            'Candidate eligibility uses offline known GT positives/negatives; not full deployment candidates.',
            'Teacher trigger measures native+DINO cosine, while KL target uses pure DINO cosine.',
            'N2 has a linear bottleneck but tanh and normalization; not strictly linear end-to-end.',
            'N3 uses native-fused ranking cross-entropy; not a full standard contrastive pretraining study.',
            'No-camera retains time; with two cameras destination is inferable from source view.',
            'No new HOTA/IDSW/FPS, no encoder runs, no sealed videos or TEST accessed.'])
    (a.out / 'attribution_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=audit['status'], paired=comparisons, changed_clauses=len(changed_rows))))


if __name__ == '__main__':
    main()

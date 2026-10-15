"""Frozen old heads on sealed DEV12; joint primary replay, never online claims."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import subprocess
import time

import torch
from scipy.optimize import linear_sum_assignment

from cv_pil_dev12_shards import NativeShardAssets
from cv_pil_dev12_assets import require
from cv_pil_native_handoff import sha256, safe_torch_load
from cv_pil_v2_model import CandidatePredictor, parameter_digest, tensorize, fused_scores

ARMS = ('C0', 'C1', 'C2', 'C3', 'C4')


def primary_solve(scores, ids, thresholds):
    require(scores.shape[1] == len(ids) == len(thresholds), 'Full candidate columns required')
    rows, columns = linear_sum_assignment(-scores.numpy())
    accepted = [None] * scores.shape[0]
    for row, col in zip(rows, columns):
        if scores[row, col] > thresholds[col]: accepted[row] = ids[col]
    return accepted


def classify(identity, label, ids):
    positive = label['certified_target_indices']; known = label['candidate_known_mask']
    if identity is None:
        if label['GT_person'] is None: return None
        if positive: return False  # known reachable candidate was rejected
        return True if all(known) else None
    i = ids.index(identity)
    return (i in positive) if known[i] else None


def retrieval(scores, ids, label):
    if not len(ids): return None, None, None
    order = torch.argsort(scores, descending=True, stable=True).tolist()
    prediction = ids[order[0]]; positives = label['certified_target_indices']
    rr = 1 / (min(order.index(i) for i in positives) + 1) if positives else None
    return prediction, rr, classify(prediction, label, ids)


def paired(rows, before, after, kind):
    a, b = before+'_'+kind, after+'_'+kind
    fixes = [r for r in rows if r[a] is False and r[b] is True]
    harms = [r for r in rows if r[a] is True and r[b] is False]
    groups = defaultdict(lambda: dict(fixes=0, harms=0))
    for category, subset in [('fixes', fixes), ('harms', harms)]:
        for r in subset: groups[str(r['person'])][category] += 1
    return dict(certified_fixes=len(fixes), certified_harms=len(harms),
                unique_fix_person_groups=len({r['person'] for r in fixes}),
                unique_harm_person_groups=len({r['person'] for r in harms}),
                net_positive_person_groups=sum(g['fixes'] > g['harms'] for g in groups.values()),
                by_person=dict(groups),
                correct_to_unknown=sum(r[a] is True and r[b] is None for r in rows),
                unknown_to_correct=sum(r[a] is None and r[b] is True for r in rows),
                fix_keys=[r['key'] for r in fixes], harm_keys=[r['key'] for r in harms])


def summarize(rows):
    methods = {}
    for arm in ARMS:
        covered = [r for r in rows if r[arm+'_rr'] is not None]
        methods[arm] = dict(queries=len(rows), positive_covered=len(covered),
            top1=sum(r[arm+'_rr'] == 1 for r in covered)/len(covered) if covered else None,
            mrr=sum(r[arm+'_rr'] for r in covered)/len(covered) if covered else None,
            top1_known_correct=sum(r[arm+'_rank'] is True for r in rows),
            top1_known_wrong=sum(r[arm+'_rank'] is False for r in rows),
            top1_unknown=sum(r[arm+'_rank'] is None for r in rows),
            primary_known_correct=sum(r[arm+'_primary'] is True for r in rows),
            primary_known_wrong=sum(r[arm+'_primary'] is False for r in rows),
            primary_unknown=sum(r[arm+'_primary'] is None for r in rows),
            primary_accepts=sum(r[arm+'_accepted_id'] is not None for r in rows),
            full_native_committed_fixes=None, full_native_committed_harms=None)
    pairs = {f'{after}_vs_{before}': {kind: paired(rows, before, after, kind) for kind in ('rank', 'primary')}
             for before, after in [('C0', 'C1'), ('C0', 'C2'), ('C0', 'C3'), ('C0', 'C4'),
                                   ('C2', 'C3'), ('C4', 'C3')]}
    for name, item in pairs.items():
        after, before = name.split('_vs_')
        item['primary_action_changes'] = sum(r[before+'_accepted_id'] != r[after+'_accepted_id'] for r in rows)
        item['rank_choice_changes'] = sum(r[before+'_rank_id'] != r[after+'_rank_id'] for r in rows)
    return dict(methods=methods, paired=pairs)


def execute(pins, output):
    require(not output.exists(), 'Refusing to overwrite prior result')
    started = time.monotonic(); torch.set_num_threads(1)
    assets = NativeShardAssets(**pins['release'])
    require(not assets.engineering_only, 'No model evaluation on prefix engineering assets')
    old = json.loads(Path(pins['frozen_v2_results']).read_text()); models = {}
    require(sha256(pins['frozen_v2_results']) == pins['frozen_v2_results_sha256'], 'V2 result SHA differs')
    for arm in ('C2', 'C3', 'C4'):
        t = old['training'][arm]; require(sha256(t['local_checkpoint']) == t['checkpoint_sha256'], 'Old head SHA differs')
        state = safe_torch_load(t['local_checkpoint'])
        require(state['arm'] == arm and state['steps'] == 300 and state['seed'] == 20261015, 'Old head contract differs')
        model = CandidatePredictor(use_camera=arm != 'C4'); model.load_state_dict(state['state_dict'], strict=True)
        require(parameter_digest(model) == t['final_parameter_sha256'], 'Old head parameters differ')
        models[arm] = model.eval()
    # Form every score/action without loading GT labels. Labels are joined only
    # after all primary calls have been scored. Same-state paired intervention.
    scored = {}; original_action_exact = True
    with torch.no_grad():
        for cid, call in assets.calls.items():
            if call['stage'] == 'run_memory_tracker': continue
            matrices = {arm: [] for arm in ARMS}
            for source in call['queries']:
                k = (12, *source); batch = tensorize(assets.actor(k))
                score = dict(C0=batch.native_scores, C1=fused_scores(batch, batch.histories))
                score.update({arm: model(batch)[0] for arm, model in models.items()})
                for arm in ARMS: matrices[arm].append(score[arm])
            if not call['queries']: continue
            matrices = {arm: torch.stack(s) for arm, s in matrices.items()}
            actions = {arm: primary_solve(s, call['track_ids'], call['thresholds']) for arm, s in matrices.items()}
            expected = [None if i < 0 else i for i in call['decision']['accepted_track_ids']]
            require(actions['C0'] == expected, 'C0 full original primary replay differs')
            for row, source in enumerate(call['queries']):
                scored[(12, *source)] = dict(scores={arm: m[row] for arm, m in matrices.items()},
                                             actions={arm: a[row] for arm, a in actions.items()})
            if len(scored) % 500 < len(call['queries']): print(json.dumps(dict(scored=len(scored))), flush=True)
    labels = assets.supervision(); records = []
    for k in assets.query_index:
        label = labels[k]; actor_meta = assets.observations[k[1:]]
        ids = assets.calls[assets.query_index[k][0]]['track_ids']; s = scored[k]
        r = dict(key=list(k), person=label['GT_person'], camera=actor_meta['camera'], frame=actor_meta['frame'],
                 time_bin50=(actor_meta['frame']-1)//50, positive_covered=bool(label['certified_target_indices']),
                 source_primary_state=label['actual_primary_state'],
                 original_bank_restored=label['source_label']['bank_restored'],
                 original_newborn=label['source_label']['newborn'])
        for arm in ARMS:
            pred, rr, outcome = retrieval(s['scores'][arm], ids, label)
            r.update({arm+'_rank_id': pred, arm+'_rr': rr, arm+'_rank': outcome,
                      arm+'_accepted_id': s['actions'][arm], arm+'_primary': classify(s['actions'][arm], label, ids)})
        records.append(r)
    result = summarize(records); result['by_scene'] = {'12': summarize(records)}
    for name, grouping in [('by_person', lambda r: str(r['person'])),
                           ('by_person_camera', lambda r: f"{r['person']}:{r['camera']}"),
                           ('by_person_camera_time50', lambda r: f"{r['person']}:{r['camera']}:{r['time_bin50']}")]:
        groups = defaultdict(list)
        for r in records:
            if r['person'] is not None: groups[grouping(r)].append(r)
        result[name] = {key: summarize(group) for key, group in groups.items()}
    p32 = result['paired']['C3_vs_C2']['primary']; p34 = result['paired']['C3_vs_C4']['primary']
    primary_opportunities = sum(r['source_primary_state'] in ('wrong_link', 'missed_known_positive') and r['positive_covered'] for r in records)
    if not primary_opportunities: status = 'INSUFFICIENT_NATIVE_OPPORTUNITY'
    elif p32['certified_fixes'] <= p32['certified_harms']: status = 'NO_UNIQUE_PREDICTIVE_EFFECT'
    elif p34['certified_fixes'] <= p34['certified_harms']: status = 'CAMERA_CONDITION_VALUE_NOT_ESTABLISHED'
    else: status = 'OFFLINE_DEVELOPMENT_POSITIVE'
    result.update(status=status, scientific_success=False, candidates='complete original native candidate columns',
                  frozen_old_checkpoint_sha256={arm: old['training'][arm]['checkpoint_sha256'] for arm in models},
                  primary_replay='one-step paired full Hungarian under identical original state; not mutated-state online',
                  original_primary_OFF_replay_exact=original_action_exact,
                  counterfactual_Bank_and_birth='NOT_REPLAYED; original actions retained only as source semantics',
                  certified_final_committed_fixes=None, certified_final_committed_harms=None,
                  source_primary_reachable_errors=primary_opportunities,
                  source_primary_states=dict(Counter(r['source_primary_state'] for r in records)),
                  unique_known_video_person_groups=len({r['person'] for r in records if r['person'] is not None}),
                  query_unknown=sum(r['person'] is None for r in records),
                  coverage_all_queries=sum(r['positive_covered'] for r in records)/len(records),
                  head_history='all strictly scene-past full references; no Top8/latest4 selection',
                  no_retraining=True, gpu_wall_hours=0, full_video_inference=False,
                  online_metrics={k: None for k in ['HOTA','AssA','DetA','CVIDF1','IDF1','MOTA','IDSW','Frag','FPS']},
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1]).decode().strip(),
                  source_sha256={f: sha256(Path(__file__).with_name(f)) for f in
                      ['run_cv_pil_v3_old_heads.py','cv_pil_dev12_shards.py','cv_pil_v2_model.py']},
                  wall_seconds=time.monotonic()-started)
    output.mkdir(parents=True)
    (output/'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (output/'QUERY_RESULTS_OFFLINE_ONLY.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(records)
    (output/'INPUT_PINS.json').write_text(json.dumps(pins, indent=2)+'\n')
    print(json.dumps(dict(status=status, queries=len(records), C3_vs_C2_primary=p32, C3_vs_C4_primary=p34)), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--pins', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); execute(json.loads(a.pins.read_text()), a.out)


if __name__ == '__main__': main()

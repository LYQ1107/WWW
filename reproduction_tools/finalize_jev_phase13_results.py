"""Finish Phase XIII only from completed native runs and actual evaluator files.

This is a reporting tool. It cannot change policies, select checkpoints, open
heldout videos, or fill missing experiments with simulated metrics.
"""
import argparse
import collections
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jev_phase13_learning import *

MAIN = ['full', 'motip', 'camel', 'set_transformer']
METRICS = ['HOTA', 'AssA', 'IDF1', 'IDSW', 'MOTA', 'Frag']
LARGE = Path('/data1/liuyeqiang/WWW_jev_phase13_runtime/20261009_v2')


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    return {'path': str(path), 'SHA256': sha(path)}


def actual_online(variant, seed, phase='formal', no_calibration=False):
    name = f'{variant}_seed{seed}' + ('_no_calibration' if no_calibration else '')
    pooled_path = OUT / f'{phase}_pooled_v1' / name / 'RESULT.json'
    pooled = read(pooled_path)
    assert pooled['status'] == 'COMPLETE' and pooled['all_six_cameras_pooled']
    videos = []
    counts = collections.Counter()
    identities = collections.Counter()
    episodes = []
    bins = collections.defaultdict(lambda: [0, 0., 0.])
    calibration_sum = collections.Counter()
    calibration_n = 0
    continuity = [0, 0]
    for video in VAL:
        root = OUT / f'{phase}_online_v1' / name / f'video{video:02d}'
        path = root / 'RESULT.json'
        r = read(path)
        assert r['status'] == 'COMPLETE' and r['actual_mutated_state_online'] and r['GTA_throw_mock']
        assert sha(r['raw_predictions']['path']) == r['raw_predictions']['SHA256']
        decision = r['decision_summary']
        assert decision['saved_logits_assignment_matches_actual_native_events']
        counts.update(decision['counts'])
        identity = read(root / 'IDENTITY_AUDIT.json')
        for key in ['known_observations', 'unknown_GT_observations', 'global_identity_wrong_observations',
                    'global_unmapped_fragment_observations', 'birth_anchor_wrong_observations',
                    'wrong_ID_duration_total_camera_frames', 'false_merge_tracks', 'extra_birth_fragments']:
            identities[key] += identity[key]
        episodes.extend(identity['wrong_ID_episodes'])
        for q in identity['identity_index_query']['queries']:
            if q['expected_cross_camera']:
                continuity[1] += 1
                continuity[0] += q['cross_camera_continuity']
        cal = decision['calibration']
        n = decision['counts'].get('MATCH_certified_rows', 0)
        if n:
            calibration_n += n
            for key in ['NLL', 'Brier']:
                assert cal[key] is not None and math.isfinite(cal[key])
                calibration_sum[key] += n * cal[key]
            for b in cal['reliability_bins']:
                entry = bins[round(b['lo'], 6)]
                entry[0] += b['count']
                entry[1] += b['count'] * b['confidence']
                entry[2] += b['count'] * b['accuracy']
        videos.append({'video': video, 'result': reference(path), 'binding': r['binding'],
                       'frames': r['frames'], 'strict_metrics': r['strict_online_metrics'],
                       'identity': r['identity_summary'], 'cross_camera': r['cross_camera'],
                       'decision': decision, 'native_counts': r['counts'],
                       'raw_predictions': r['raw_predictions'], 'trained': r['trained'],
                       'stage2_latency_with_audit_instrumentation_ms': r['stage2_latency_with_audit_instrumentation_ms']})
    reliability = [{'lo': lo, 'count': n, 'confidence': c / n, 'accuracy': a / n}
                   for lo, (n, c, a) in sorted(bins.items()) if n]
    durations = [e['duration_frames'] for e in episodes]
    ratio = lambda a, b: counts[a] / counts[b] if counts[b] else None
    return {'variant': variant, 'seed': seed, 'phase': phase, 'pooled_result': reference(pooled_path),
            'tracking': pooled['strict_pooled_TrackEval'], 'videos': videos, 'decision_counts': dict(counts),
            'normal_association_keep_rate': ratio('normal_supported_correct', 'normal_supported_rows'),
            'Gallery_contamination_fraction': ratio('contaminated_Gallery_IDs', 'known_Gallery_IDs'),
            'identity': dict(identities), 'cross_camera_continuity': continuity[0] / continuity[1] if continuity[1] else None,
            'cross_camera_supported_GT_identities': continuity[1],
            'error_duration': {'definition': 'consecutive actually observed wrong first-GT-anchor ID in each camera; gaps/end are right-censored; no counterfactual recovery time',
                               'episode_count': len(episodes), 'right_censored_episodes': sum(e['right_censored'] for e in episodes),
                               'total_camera_frames': sum(durations),
                               'quantiles_frames': dict(zip(['p50', 'p90', 'max'], map(float, np.quantile(durations, [.5, .9, 1])))) if durations else {'p50': 0., 'p90': 0., 'max': 0.}},
            'calibration': {'scope': 'MATCH conditional certified options only; contaminated/UNKNOWN options excluded only in offline audit, never in actor',
                            'rows': calibration_n, **{k: calibration_sum[k] / calibration_n if calibration_n else None for k in ['NLL', 'Brier']},
                            'ECE': sum(b['count'] * abs(b['confidence'] - b['accuracy']) for b in reliability) / calibration_n if calibration_n else None,
                            'reliability_bins': reliability, 'risk_coverage': 'actual per-video curves retained in videos[].decision.calibration; no mean curve presented as pooled ranking'}}


def aggregate_onpolicy():
    cases = []
    for variant in MAIN:
        for seed in SEEDS:
            path = OUT / 'onpolicy_training_v1' / variant / f'seed{seed}' / 'RESULT.json'
            manifest_path = LARGE / 'onpolicy_native_v1' / variant / f'seed{seed}' / 'MANIFEST.json'
            if path.exists():
                r = read(path)
                assert r['status'] == 'COMPLETE' and r['actual_updates'] == 4000 and r['total_updates'] == 24000
                assert sha(r['checkpoint']['path']) == r['checkpoint']['SHA256']
                online = actual_online(variant, seed, 'onpolicy')
                before = {r['video']: r['summary'] for r in r['before_online_TRAIN']}
                after = {r['video']: r['summary'] for r in r['after_online_TRAIN']}
                paired = [{ 'video': v, 'before': before[v], 'after': after[v],
                            'wrong_anchor_observation_delta': after[v]['birth_anchor_wrong_observations'] - before[v]['birth_anchor_wrong_observations'],
                            'extra_birth_fragments_delta': after[v]['extra_birth_fragments'] - before[v]['extra_birth_fragments'],
                            'wrong_ID_duration_total_camera_frames_delta': after[v]['wrong_ID_duration_total_camera_frames'] - before[v]['wrong_ID_duration_total_camera_frames']}
                          for v in TRAIN]
                cases.append({'status': 'COMPLETE', 'variant': variant, 'seed': seed, 'training_result': reference(path),
                              'training': r, 'own_state_manifest': reference(manifest_path), 'own_state_counts': read(manifest_path)['counts'],
                              'paired_first256_TRAIN': paired, 'online_development': online})
            else:
                full = read(OUT / 'formal_training_v1/full' / f'seed{seed}' / 'RESULT.json')
                if full['TRAIN_FIXED_AUDIT_SUBSET']['certified_accuracy'] < .90:
                    reason = 'frozen Full normal-association TRAIN gate did not reach0.90'
                    evidence = reference(OUT / 'formal_training_v1/full' / f'seed{seed}' / 'RESULT.json')
                else:
                    manifest = read(manifest_path)
                    assert manifest['status'] == 'FAIL' and manifest['counts'].get('normal_active_positive', 0) < 5000
                    reason = 'actual own-state reliable positive support below frozen5000; fine-tune forbidden'
                    evidence = reference(manifest_path)
                cases.append({'status': 'NOT_RUN', 'variant': variant, 'seed': seed, 'reason': reason, 'evidence': evidence, 'metrics': None})
    report = {'status': 'COMPLETE_WITH_QUALIFICATION_LIMITS' if any(c['status'] == 'NOT_RUN' for c in cases) else 'COMPLETE',
              'binding': binding(), 'protocol': reference(REPORTS / 'ONPOLICY_PROTOCOL.json'), 'cases': cases,
              'rounds_max': 1, 'comparison_budget': '4000 extra updates,24000 total, separate from20000 primary',
              'evaluation_scope': 'paired actual first256 scene frames of all TRAIN videos; observed error episodes are censored; full development17/18/19 afterward',
              'GT_online_inputs': False, 'UNKNOWN_not_negative': True, 'heldout': 'SEALED'}
    save(REPORTS / 'ON_POLICY_TRAINING.json', report)
    return report


def main():
    protect()
    supervised = read(REPORTS / 'SUPERVISED_TRAINING.json')
    comparison = read(REPORTS / 'BASELINE_COMPARISON.json')
    assert supervised['status'] == comparison['status'] == 'COMPLETE'
    assert len(supervised['results']) == len(comparison['results']) == len(VARIANTS) * len(SEEDS)
    cases = [actual_online(v, s) for v in VARIANTS for s in SEEDS]
    cosine = actual_online('cosine', 20261009)
    no_calibration = [actual_online('full', s, no_calibration=True) for s in SEEDS]
    calibration_pairs = []
    for other in no_calibration:
        original = next(c for c in cases if c['variant'] == 'full' and c['seed'] == other['seed'])
        equal = all(a['raw_predictions']['SHA256'] == b['raw_predictions']['SHA256'] for a, b in zip(original['videos'], other['videos']))
        calibration_pairs.append({'seed': other['seed'], 'all_three_actual_raw_prediction_SHA_equal': equal,
                                  'tracking_delta': {k: other['tracking'][k] - original['tracking'][k] for k in METRICS},
                                  'calibrated': original['calibration'], 'T1': other['calibration'],
                                  'no_calibration_result': other['pooled_result']})
    lifecycle = [{'name': 'MATCH_ONLY', 'status': 'COMPLETE', 'metrics_source': 'all formal cases; shared untrained cosine REACT and native WRITE fallback'},
                 {'name': 'MATCH_PLUS_REACTIVATION', 'status': 'NOT_RUN', 'reason': 'only9 natural stale-positive TRAIN rows, below frozen100 across3 videos; no invented detections/candidates', 'metrics': None},
                 {'name': 'THREE_LIFECYCLE', 'status': 'NOT_RUN', 'reason': 'stale support unqualified and0 WRITE/KEEP labels; future-GMT labels forbidden', 'metrics': None}]
    official = read(REPORTS / 'OFFICIAL_MATLAB_CROSSVIEW.json')
    assert official['status'] == 'COMPLETE'
    assert sum(c['phase']=='formal' for c in official['cases']) == 40
    assert all(c['status']=='COMPLETE' for c in official['cases'])
    save(REPORTS / 'ONLINE_VALIDATION.json', {'status': 'COMPLETE', 'binding': binding(), 'cases': cases, 'B1': cosine,
                                             'strict_pooled_not_video_average': True, 'official_CVIDF1_CVMA': {'status':'COMPLETE','report':reference(REPORTS/'OFFICIAL_MATLAB_CROSSVIEW.json'),'environment_audit':reference(REPORTS/'MATLAB_ENVIRONMENT_AUDIT.json')},
                                             'lifecycle': lifecycle, 'development_preexposed': True, 'heldout': 'SEALED'})
    summaries = comparison['strict_seed_summary']
    full = summaries['full']
    differences = {}
    for variant in VARIANTS:
        differences[variant] = {k: summaries[variant][k]['mean'] - full[k]['mean'] for k in METRICS}
    save(REPORTS / 'ARCHITECTURE_ABLATION.json', {'status': 'COMPLETE_WITH_LIFECYCLE_LIMITS', 'binding': binding(),
                                               'common_updates': 20000, 'seeds': SEEDS, 'strict_seed_summary': summaries,
                                               'variant_minus_Full_mean': differences, 'no_probability_calibration': calibration_pairs,
                                               'lifecycle': lifecycle, 'No_Typed_Head_interpretation': 'NOT_IDENTIFIABLE lifecycle benefit under MATCH-only supervision; measured MATCH-head parameter-sharing diagnostic only',
                                               'permutation_scope': 'neural logits/probabilities and opaque-ID renaming tested; discrete exact Hungarian ties may lack a unique assignment',
                                               'no_shared_state': {'status': 'NOT_RUN', 'reason': 'extra code variant outside frozen twelve-model formal list', 'metrics': None}})
    own = aggregate_onpolicy()
    efficiencies = []
    for variant in ['cosine'] + VARIANTS:
        path = OUT / 'live_efficiency_v1' / variant / 'seed20261009' / 'RESULT.json'
        r = read(path)
        assert r['status'] == 'COMPLETE' and r['live_real_images'] and r['perception_cache_hits'] == 0
        efficiencies.append({'result': reference(path), 'values': r})
    full_efficiency = next(e['values'] for e in efficiencies if e['values']['variant'] == 'full')
    save(REPORTS / 'EFFICIENCY.json', {'status': 'COMPLETE', 'binding': binding(), 'cases': efficiencies,
                                      'representative_seed': 20261009, 'scope': 'isolated one-V100 actual image I/O/detector/VFCE/native first256-frame run; eight warmup frames; no GT/journals/evaluator in timed path',
                                      'formal_36_training_profiles': [{'variant': r['variant'], 'seed': r['seed'], 'capacity': r['result']['capacity'], 'peak_training_VRAM_MiB': r['result']['peak_training_VRAM_MiB']} for r in supervised['results']],
                                      'loaded_original_GTA_dormant': True, 'stage2_and_full_FPS_distinguished': True})
    b1 = cosine['tracking']
    g5 = full['HOTA']['mean'] > b1['HOTA'] and full['AssA']['mean'] > b1['AssA'] and full['IDSW']['mean'] <= 1.25 * b1['IDSW']
    styles = ['motip', 'camel', 'set_transformer']
    g6 = all(r['result']['actual_updates'] == 20000 for r in supervised['results'])
    caps = {r['variant']: r['result']['capacity']['registered_parameters'] for r in supervised['results'] if r['variant'] in MAIN}
    g6 = g6 and all(.9 <= caps['full'] / caps[v] <= 1.1 for v in styles)
    seed_deltas = []
    for s in SEEDS:
        sf = next(c for c in cases if c['variant'] == 'full' and c['seed'] == s)
        for v in styles:
            st = next(c for c in cases if c['variant'] == v and c['seed'] == s)
            seed_deltas.append({'seed': s, 'style': v, **{k: sf['tracking'][k] - st['tracking'][k] for k in METRICS}})
    reduced = [v for v in VARIANTS if v not in MAIN and differences[v]['HOTA'] < 0]
    g7 = (full['HOTA']['mean'] >= max(summaries[v]['HOTA']['mean'] for v in styles) + .5
          and full['AssA']['mean'] >= max(summaries[v]['AssA']['mean'] for v in styles) + .5
          and all(d['HOTA'] >= 0 for d in seed_deltas) and len(reduced) >= 2)
    gates = {
        'G0': {'status': 'PASS_WITH_PRETRAIN_EXPOSURE_LIMIT', 'evidence': reference(REPORTS / 'STAGE1_CHECKPOINT_AUDIT.json')},
        'G1': {'status': 'PASS', 'evidence': reference(REPORTS / 'GTA_FREE_CONTRACT.json'), 'all_actual_online_GTA_throw_mocks': True},
        'G2': {'status': 'PASS_MATCH_ONLY', 'evidence': reference(REPORTS / 'DENSE_DATASET_MANIFEST.json'), 'REACT': 'UNQUALIFIED', 'MEMORY': 'UNQUALIFIED'},
        'G3': {'status': read(REPORTS / 'TINY_LEARNABILITY.json')['status'], 'evidence': reference(REPORTS / 'TINY_LEARNABILITY.json')},
        'G4': {'status': read(REPORTS / 'NATIVE_PARITY.json')['status'], 'evidence': reference(REPORTS / 'NATIVE_PARITY.json'), 'repeated_recovery_fixed': reference(REPORTS / 'NATIVE_RECYCLING_BUG_AUDIT.json')},
        'G5': {'status': 'PASS' if g5 else 'FAIL', 'Full_mean': {k: full[k]['mean'] for k in METRICS}, 'B1': b1,
               'failure_reasons': [name for name, passed in [('HOTA must exceed B1', full['HOTA']['mean'] > b1['HOTA']), ('AssA must exceed B1', full['AssA']['mean'] > b1['AssA']), ('IDSW must not exceed1.25xB1', full['IDSW']['mean'] <= 1.25 * b1['IDSW'])] if not passed],
               'severe_collapse': full['HOTA']['mean'] < b1['HOTA'] - 5 or full['IDSW']['mean'] > 2 * b1['IDSW']},
        'G6': {'status': 'PASS' if g6 else 'FAIL', 'registered_parameters': caps, 'same_data_labels_candidates_native_solver': True, 'updates': 20000, 'seeds': SEEDS},
        'G7': {'status': 'EXPLORATORY_GO' if g7 else 'NO_GO', 'Full_minus_style_by_seed': seed_deltas, 'HOTA_reducing_ablations': reduced,
               'inference_scope': 'frozen exploratory threshold on3 historically preexposed development scenes; no statistical population superiority claim'},
        'G8': {'status': 'PASS' if full_efficiency['budget_PASS'] else 'FAIL', 'values': full_efficiency},
        'G9': {'status': 'NO_GO', 'reason': 'inherited true Stage1 was trained on all24 TRAIN videos including20/21/22; independent feature-level heldout authorization fails', 'heldout': 'SEALED'}
    }
    report = {'status': 'NO_GO_HELDOUT', 'execution': 'COMPLETE_WITH_SCIENTIFIC_QUALIFICATION_LIMITS', 'binding': binding(), 'gates': gates,
              'claim_scope': 'MATCH-only controller on historically preexposed development; typed lifecycle and independent heldout superiority are not established',
              'main_claim_supported': g5 and g6 and g7 and full_efficiency['budget_PASS'], 'official_TEST': False, 'Full24': False,
              'lifecycle': lifecycle, 'onpolicy_status': own['status'], 'environment': reference(REPORTS / 'RUN_ENVIRONMENT.json')}
    save(REPORTS / 'FINAL_GO_NO_GO.json', report)
    plot_results(summaries, cases)
    write_report(report, summaries, cosine, own)
    goal = read(REPORTS / 'FINAL_GOAL.json')
    goal.update(status='COMPLETE_WITH_SCIENTIFIC_LIMITS', final=reference(REPORTS / 'FINAL_GO_NO_GO.json'), binding=binding())
    save(REPORTS / 'FINAL_GOAL.json', goal)
    print('PHASE13_FINAL_COMPLETE', {k: v['status'] for k, v in gates.items()}, flush=True)


def plot_results(summaries, cases):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric in zip(axes, ['HOTA', 'AssA']):
        for i, v in enumerate(VARIANTS):
            values = [c['tracking'][metric] for c in cases if c['variant'] == v]
            axis.scatter([i] * len(values), values, s=16)
        axis.set_xticks(range(len(VARIANTS)))
        axis.set_xticklabels(VARIANTS, rotation=75, fontsize=6)
        axis.set_ylabel(f'Strict pooled {metric}')
        axis.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(REPORTS / 'FORMAL_TRACKING_SEEDS.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for v in MAIN:
        c = next(c for c in cases if c['variant'] == v and c['seed'] == 20261009)
        bins = c['calibration']['reliability_bins']
        axes[0].plot([b['confidence'] for b in bins], [b['accuracy'] for b in bins], marker='.', label=v)
        # These are explicitly per-video curves, not a fabricated pooled rank.
        for video in c['videos']:
            risk = video['decision']['calibration']['risk_coverage']
            axes[1].plot([p['coverage'] for p in risk], [p['risk'] for p in risk], label=f"{v} video{video['video']}")
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlabel='Conditional confidence', ylabel='Certified-choice accuracy', title='Pooled reliability, seed20261009')
    axes[1].set(xlabel='Coverage', ylabel='Risk', title='Actual per-video risk/coverage')
    for a in axes:
        a.grid(alpha=.2)
        a.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(REPORTS / 'CALIBRATION_AND_RISK.png', dpi=160)
    plt.close(fig)


def write_report(report, summaries, cosine, own):
    lines = ['# Phase XIII final research report', '',
             'All primary models below use verified true Stage1 VFCE1024, full natural TRAIN12/13/14/16, common legal candidates/native executor, and fresh20,000-update checkpoints. Three seeds are evaluated on all complete development17/18/19 videos. Scores are actual pooled TrackEval over six cameras, followed by seed means. These development scenes were used historically and were exposed during inherited Stage1 pretraining.', '',
             '| Method | HOTA | AssA | IDF1 | IDSW | MOTA | Frag |', '|---|---:|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        lines.append('| ' + v + ' | ' + ' | '.join(f'{summaries[v][k]["mean"]:.3f}' for k in METRICS) + ' |')
    lines.append('| B1 cosine | ' + ' | '.join(f'{cosine["tracking"][k]:.3f}' for k in METRICS) + ' |')
    original = read(REPORTS / 'GMT_BASELINE_FREEZE.json')['GMT_OFF_metrics']['strict_online']
    lines.append('| Original GMT separate full system | ' + ' | '.join(f'{original[k]:.3f}' for k in METRICS) + ' |')
    lines.extend(['', '## Bounded own-state round', '',
                  'The following comparison uses the same completed eligible seeds before and after the extra4k updates. Wrong-anchor observations count actual errors relative to each identity first GT anchor; they must be read together with births/fragments and full tracking metrics. Prefix error duration is censored and is not a counterfactual propagation estimate.', '',
                  '| Method | Eligible completed seeds | 20k HOTA paired | 24k HOTA | 20k IDSW paired | 24k IDSW | TRAIN wrong-anchor before | after | TRAIN extra births before | after |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for variant in MAIN:
        cc = [c for c in own['cases'] if c['variant'] == variant and c['status'] == 'COMPLETE']
        if not cc:
            lines.append(f'| {variant} | 0 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |')
            continue
        before_tracking = [read(OUT/'formal_pooled_v1'/f'{variant}_seed{c["seed"]}'/'RESULT.json')['strict_pooled_TrackEval'] for c in cc]
        avg = lambda values: float(np.mean(values))
        before_errors = [sum(r['before']['birth_anchor_wrong_observations'] for r in c['paired_first256_TRAIN']) for c in cc]
        after_errors = [sum(r['after']['birth_anchor_wrong_observations'] for r in c['paired_first256_TRAIN']) for c in cc]
        before_fragments = [sum(r['before']['extra_birth_fragments'] for r in c['paired_first256_TRAIN']) for c in cc]
        after_fragments = [sum(r['after']['extra_birth_fragments'] for r in c['paired_first256_TRAIN']) for c in cc]
        values = [avg([r['HOTA'] for r in before_tracking]), avg([c['online_development']['tracking']['HOTA'] for c in cc]),
                  avg([r['IDSW'] for r in before_tracking]), avg([c['online_development']['tracking']['IDSW'] for c in cc]), avg(before_errors), avg(after_errors), avg(before_fragments), avg(after_fragments)]
        lines.append('| '+variant+' | '+str(len(cc))+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines.extend(['', 'Original GMT uses Stage2-trained perception/RPCE and is a separate strong full-system comparator. MOTIP-style/CAMEL-style are controlled adapters, not complete official framework reproductions.', '',
                  'Native state parity exposed a restored-ID recycling bug before primary training. Old data/models/results remain archived. The correction adds a recovered ID back to the real possible-ID set; the real GTA-throw lifecycle test now recovers the same ID twice and verifies full/resumed state through106 frames. All primary models use the regenerated v3 corpus. A dormant-state issue in the Set control was also repaired before training; every shared-state/ordinary decoder block has nonzero supervised gradients.', '',
                  'Tiny/Pilot use frozen examples and budgets; formal primary checkpoint is LAST at20k. All GT labels are offline and uncertain/contaminated candidate histories remain UNKNOWN. REACT has only9 natural clean TRAIN positives and MEMORY has0 WRITE/KEEP labels, so learned MATCH+REACT and three-lifecycle experiments are NOT_RUN. Common cosine recovery/native WRITE remain explicit fallbacks. The NoTyped result cannot identify typed lifecycle benefit under MATCH-only supervision.', '',
                  f'On-policy round status: {own["status"]}. Its frozen bound is first256 frames of each of four TRAIN videos, one4k update round per eligible main model/seed, with its own actual mutated histories. Total24k outcomes are kept separate from20k; before/after observed wrong-ID counts and censored duration are in ON_POLICY_TRAINING.json.', '',
                  'No-calibration runs use the exact same Full checkpoints with T=1 on all three development videos. Actual raw-prediction hashes and metric deltas are in ARCHITECTURE_ABLATION.json; positive uniform temperature should preserve a unique maximum-sum assignment, and calibration benefit is restricted to certified MATCH probabilities.', '',
                  'Actual isolated live image/detector/VFCE/native measurements, executed matrix-multiply cost, latency p50/p95 and full VRAM are in EFFICIENCY.json. Cache-backed validation timing is never called full FPS. Native MATLAB R2020a was installed from the user-supplied ISO, and untouched official evaluator/MEX passed analytic perfect/miss/false-positive/ID-split fixtures. Actual native official CVIDF1/CVMA over all frozen raw and canonical cases are in OFFICIAL_MATLAB_CROSSVIEW.json. The separately labelled Python TrackEval adaptation remains available for comparison.', '',
                  '| Gate | Result |', '|---|---|'])
    lines.extend(f'| {k} | {v["status"]} |' for k, v in report['gates'].items())
    lines.extend(['', f'Frozen exploratory independent-value gate: {report["gates"]["G7"]["status"]}. These three preexposed scenes do not establish population-level significance. See exact per-seed paired deltas and structural ablation means in FINAL_GO_NO_GO.json.', '',
                  'Heldout20/21/22 remain sealed: inherited Stage1 already saw every TRAIN video, so an independent full-system heldout claim is invalid. Full24 and official TEST remain unauthorized. A genuinely unseen feature-level evaluation requires an explicitly authorized perception retraining/split redesign before reopening heldout.', '',
                  'Large datasets/checkpoints/predictions/segmented native logs remain on the server. Git contains only source, protocols, compact results/plots and SHA references. All PhaseV–XII scientific assets and failures remain intact. RUN_ENVIRONMENT.json plus each run binding records actual source/data/weights/config/seed provenance.', ''])
    (ROOT / 'docs/JEV_PHASE13_FINAL_RESEARCH_REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

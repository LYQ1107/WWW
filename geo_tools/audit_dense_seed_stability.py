"""Keep TRAIN-head variation separate from fixed-head inference-RNG trials."""
from common import *
from collections import Counter
from audit_camera_control import verify_pure_gmt
from audit_dense_geometry_ablation import shared_options
from audit_suite_settings import signature


SEEDS = [20261014, 20261015, 20261016]


def check_trial(run, reference, head, inference_seed):
    if run['seed'] != inference_seed:
        raise ValueError('actual inference RNG differs from declared trial')
    if inference_seed != SEEDS[0] and run.get('explicit_inference_seed') != inference_seed:
        raise ValueError('explicit model/stream seed wrapper evidence missing')
    if run['head_checkpoint_sha256'] != head['checkpoint_sha256']:
        raise ValueError('wrong TRAIN head; head seed is not inference seed')
    if sha256(run['geo_options']['HEAD_PATH']) != head['checkpoint_sha256']:
        raise ValueError('changed head artifact')
    signatures = []
    options = []
    for value in (run, reference):
        setting, verified = signature(value)
        if not verified:
            raise ValueError('unverified source config defaults')
        for key in ('seed', 'head_checkpoint_sha256'):
            setting.pop(key)
        signatures.append(setting)
        resolved = shared_options(value)
        resolved.pop('HEAD_PATH')
        resolved['GEOMETRY_WEIGHT'] = value['geo_options']['GEOMETRY_WEIGHT']
        options.append(resolved)
    if signatures[0] != signatures[1] or options[0] != options[1]:
        raise ValueError('settings differ beyond the declared seed/head factors')
    if not run['causal_online'] or not run['full_video'] or run['geo_failures']:
        raise ValueError('complete successful causal trial required')
    if run['total_vggt_calls'] != run['images']:
        raise ValueError('dense encoder cost does not match all input images')


def scene1_diagnostic(heads, artifacts):
    """A fixed scene already declared in every suite, never a suite substitute."""
    source = ROOT/'manifests/causal_B0_strict_suite_v1_sources.json'
    baseline_prediction = Path(json.loads(source.read_text())['video_predictions']['1'])
    baseline_manifest = baseline_prediction.parent/'manifest.json'
    baseline_metric = baseline_prediction.with_suffix('.metrics.json')
    original = json.loads(baseline_manifest.read_text())
    base = json.loads(baseline_metric.read_text())
    verify_pure_gmt(original)
    def detectors(path):
        return Counter((row['image_id'], row['category_id'], row['score'], *row['bbox']) for row in json.loads(path.read_text()))
    baseline_detectors = detectors(baseline_prediction)
    rows, pending = [], []
    for method in ('B5', 'B7'):
        primary = 'causal_B5_dense_seed20261014_strict_suite_v1' if method == 'B5' else 'causal_B7_fixed_k1_strict_suite_v1'
        reference_suite = json.loads((ROOT/'results'/f'{primary}_metrics.json').read_text())
        capture = reference_suite['run_manifests']['1']
        if sha256(capture['path']) != capture['sha256']:
            raise ValueError('changed primary scene reference')
        reference = json.loads(Path(capture['path']).read_text())
        for seed in SEEDS:
            suite = primary if seed == SEEDS[0] else f'causal_{method}_dense_seed{seed}_strict_suite_v1'
            spec = ROOT/'manifests'/f'{suite}_sources.json'
            prediction = Path(json.loads(spec.read_text())['video_predictions']['1'])
            metric_path = prediction.with_suffix('.metrics.json')
            manifest_path = prediction.parent/'manifest.json'
            if not manifest_path.exists() or not metric_path.exists():
                pending.append(dict(method=method, training_head_seed=seed))
                continue
            run = json.loads(manifest_path.read_text())
            metric = json.loads(metric_path.read_text())
            check_trial(run, reference, heads[seed], SEEDS[0])
            for key in ('checkpoint_sha256', 'config_sha256', 'annotation_sha256', 'seed', 'video_id', 'images', 'synchronized_groups'):
                if original[key] != run[key]:
                    raise ValueError('confounded scene1 comparison: '+key)
            for info, pred, stats in ((original, baseline_prediction, base), (run, prediction, metric)):
                if stats['status'] != 'PASS' or not stats['full_video'] or not stats['causal_online'] or stats['video_id'] != 1 or stats['allow_duplicate_gt']:
                    raise ValueError('complete strict causal scene1 metric required')
                if sha256(pred) != stats['prediction_sha256'] or sha256(pred) != info['predictions_sha256']:
                    raise ValueError('changed scene1 predictions')
                for path, digest in stats['raw_gt_sha256'].items():
                    raw_gt = Path(path) if Path(path).is_absolute() else DATA/'train'/path/'gt/gt.txt'
                    if sha256(raw_gt) != digest:
                        raise ValueError('changed scene1 raw GT')
            if detectors(prediction) != baseline_detectors:
                raise ValueError('scene1 detector membership differs')
            rows.append(dict(method=method, training_head_seed=seed, inference_seed=SEEDS[0],
                metrics=metric['metrics'], paired_metric_delta={key: metric['metrics'][key]-base['metrics'][key] for key in base['metrics']},
                prediction=str(prediction), prediction_sha256=sha256(prediction), exact_detector_membership=True,
                actual_vggt_calls=run['total_vggt_calls'], numerical_geometry_gates=run['geometry_valid_calls']))
            artifacts.update({str(p): sha256(p) for p in (spec, prediction, manifest_path, metric_path)})
    artifacts.update({str(p): sha256(p) for p in (source, baseline_prediction, baseline_manifest, baseline_metric)})
    parity = []
    for seed in SEEDS:
        pair = [next((row for row in rows if row['method'] == method and row['training_head_seed'] == seed), None) for method in ('B5', 'B7')]
        if all(pair):
            parity.append(dict(training_head_seed=seed, exact_prediction_file_bytes=pair[0]['prediction_sha256'] == pair[1]['prediction_sha256']))
    return dict(full_strict_suite=False, video_id=1, baseline_metrics=base['metrics'], rows=rows, pending=pending,
        feature_joint_prediction_parity=parity,
        scope='fixed scene1 complete-video TRAIN-head observations at inference seed20261014; does not complete any missing five-scene trial')


def main():
    split = ROOT/'manifests/geo_split_v1.json'
    strict = sorted(json.loads(split.read_text())['strict_development_video_ids'])
    head_path = ROOT/'results/low_rank_three_backbones_train_v1_manifest.json'
    heads = {row['seed']: row for row in json.loads(head_path.read_text())['results'] if row['backbone'] == 'vggt'}
    relation_path = ROOT/'results/identity_relations_matrix_v2.json'
    relation = json.loads(relation_path.read_text())
    plans = [ROOT/'manifests'/name for name in ('dense_stability_parallel_v2.json', 'primary_vggt_inference_seed_pairs_v1.json')]
    if set(json.loads(plans[1].read_text())['inference_seeds']) != set(SEEDS[1:]):
        raise ValueError('inference RNG trials differ from the declared plan')
    artifacts = {str(p): sha256(p) for p in (split, head_path, relation_path, *plans)}
    groups = []
    for factor, method in [('training_head_seed', 'B5'), ('training_head_seed', 'B7'), ('inference_rng_seed', 'B7')]:
        primary = 'causal_B5_dense_seed20261014_strict_suite_v1' if method == 'B5' else 'causal_B7_fixed_k1_strict_suite_v1'
        reference_path = ROOT/'results'/f'{primary}_metrics.json'
        reference = json.loads(reference_path.read_text())
        artifacts[str(reference_path)] = sha256(reference_path)
        rows, pending = [], []
        for seed in SEEDS:
            head_seed = seed if factor == 'training_head_seed' else SEEDS[0]
            inference_seed = seed if factor == 'inference_rng_seed' else SEEDS[0]
            suite = primary if seed == SEEDS[0] else (f'causal_{method}_dense_seed{seed}_strict_suite_v1' if factor == 'training_head_seed' else f'causal_B7_primary_head_inference_seed{seed}_strict_suite_v1')
            baseline = 'causal_B0_strict_suite_v1' if inference_seed == SEEDS[0] else f'causal_B0_seed{inference_seed}_strict_suite_v1'
            paths = [ROOT/'results'/f'{tag}_metrics.json' for tag in (baseline, suite)]
            paired = next((row for row in relation['rows'] if row['suite'] == suite), None)
            if any(not p.exists() for p in paths) or paired is None:
                pending.append(dict(training_head_seed=head_seed, inference_seed=inference_seed, suite=suite,
                                    reason='complete native suite or paired identity audit unavailable'))
                continue
            base, candidate = [json.loads(p.read_text()) for p in paths]
            if any(d['status'] != 'PASS' or not d['full_video'] or not d['causal_online'] or d['video_ids'] != strict for d in (base, candidate)):
                raise ValueError('all five strict causal scenes required')
            if paired['protocol'] != 'causal' or paired['CVIDF1'] != candidate['metrics']['CVIDF1'] or paired['IDSW'] != candidate['metrics']['IDSW']:
                raise ValueError('identity and metric scopes differ')
            if paired['added_detections'] or paired['removed_detections']:
                raise ValueError('detector membership changed')
            audit_path = Path(paired['source'])
            if sha256(audit_path) != paired['source_sha256']:
                raise ValueError('changed paired identity audit')
            audit = json.loads(audit_path.read_text())
            if not audit['full_strict_suite'] or audit['status'] != 'PASS':
                raise ValueError('complete paired audit required')
            scenes = []
            for scene in strict:
                captures = [d['run_manifests'][str(scene)] for d in (base, candidate, reference)]
                runs = []
                for capture in captures:
                    if sha256(capture['path']) != capture['sha256']:
                        raise ValueError('changed inference manifest')
                    runs.append(json.loads(Path(capture['path']).read_text()))
                original, actual, ref = runs
                verify_pure_gmt(original)
                check_trial(actual, ref, heads[head_seed], inference_seed)
                for key in ('checkpoint_sha256', 'config_sha256', 'annotation_sha256', 'seed', 'video_id', 'images', 'synchronized_groups'):
                    if original[key] != actual[key]:
                        raise ValueError('unpaired model/data/inference RNG: '+key)
                pair = next(row for row in audit['rows'] if row['video_id'] == scene)
                for index, key in enumerate(('baseline_prediction', 'method_prediction')):
                    prediction = pair[key]
                    digest = sha256(prediction)
                    if digest != pair['prediction_sha256'][index] or digest != runs[index]['predictions_sha256'] or digest != (base, candidate)[index]['prediction_sha256'][prediction]:
                        raise ValueError('paired predictions differ from evaluated artifacts')
                scenes.append(dict(video_id=scene, actual_vggt_calls=actual['total_vggt_calls'],
                                   numerical_geometry_gates=actual['geometry_valid_calls'], manifest=captures[1]))
            for path, digest in audit['raw_gt_sha256'].items():
                if sha256(path) != digest:
                    raise ValueError('changed raw GT')
            rows.append(dict(suite=suite, training_head_seed=head_seed, inference_seed=inference_seed,
                head_checkpoint_sha256=heads[head_seed]['checkpoint_sha256'], baseline_metrics=base['metrics'], candidate_metrics=candidate['metrics'],
                paired_metric_delta={key: candidate['metrics'][key]-base['metrics'][key] for key in base['metrics']},
                cross_camera_output_pair_fixes=paired['cross_camera_baseline_wrong_method_correct'],
                cross_camera_output_pair_harms=paired['cross_camera_baseline_correct_method_wrong'],
                exact_detector_membership=True, scenes=scenes))
            artifacts.update({str(p): sha256(p) for p in [*paths, audit_path]})
        delta = [row['paired_metric_delta']['CVIDF1'] for row in rows]
        groups.append(dict(factor=factor, method=method, completed_trials=len(rows), required_trials=3,
                           complete=len(rows) == 3, rows=rows, pending=pending,
                           observed_paired_cvidf1_delta_range=[min(delta), max(delta)] if delta else None))
    scene1 = scene1_diagnostic(heads, artifacts)
    report = dict(status='COMPLETE_DECLARED_SEED_FACTORS' if all(group['complete'] for group in groups) else 'PARTIAL_DECLARED_SEED_FACTORS',
        groups=groups, source_artifacts=artifacts,
        scene1_training_head_diagnostic=scene1,
        scope='complete strict5 dense B5/B7 TRAIN-head variation at fixed inference RNG20261014, and dense B7 fixed-primary-head20261014 inference RNG variation; paired causal baseline at the same inference seed',
        limitations='primary B7 trial is shared between factors, not an independent replicate; incomplete factors do not prove stability; three trials are descriptive, without confidence intervals, seed selection, stable speed claims, unique VGGT/geometry attribution or internal Hungarian decision counts', provenance=provenance())
    save_json(ROOT/'results/causal_dense_seed_stability_v1.json', report)
    text = ['# Dense causal seed-factor audit', '', report['scope']+'.', '', report['limitations']+'.', '',
            '| Factor | Method | TRAIN head seed | Actual inference seed | Paired CVIDF1 delta (pp) | IDSW delta | Cross-camera output-pair fixes/harms |',
            '|---|---|---:|---:|---:|---:|---:|']
    for group in groups:
        for row in group['rows']:
            text.append(f"| {group['factor']} | {group['method']} | {row['training_head_seed']} | {row['inference_seed']} | {row['paired_metric_delta']['CVIDF1']:+.6f} | {row['paired_metric_delta']['IDSW']:+d} | {row['cross_camera_output_pair_fixes']}/{row['cross_camera_output_pair_harms']} |")
    text += ['', 'Missing full trials remain pending; partial scene results are not substituted for native complete-suite evaluation:', '']
    for group in groups:
        text.append(f"- {group['factor']} / {group['method']}: {group['completed_trials']}/3 complete; pending "+', '.join(row['suite'] for row in group['pending'])+'.')
    text += ['', 'Separate fixed scene1 complete-video observations (full strict suite: false):', '',
             '| Method | TRAIN head seed | Inference seed | CVIDF1 | Paired delta (pp) | IDSW | VGGT calls | Numerical geometry gates |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in scene1['rows']:
        text.append(f"| {row['method']} | {row['training_head_seed']} | {row['inference_seed']} | {row['metrics']['CVIDF1']:.6f} | {row['paired_metric_delta']['CVIDF1']:+.6f} | {row['metrics']['IDSW']} | {row['actual_vggt_calls']} | {row['numerical_geometry_gates']} |")
    text += ['', f"Scene1 causal baseline CVIDF1={scene1['baseline_metrics']['CVIDF1']:.6f}, IDSW={scene1['baseline_metrics']['IDSW']}. Head-seed variation here does not establish a complete-DEV aggregate or an inference-RNG effect. No selection, retraining or coefficient change follows these observations."]
    text += ['', 'Scene1 feature/joint prediction-file byte equality by TRAIN head: '+', '.join(f"{row['training_head_seed']}={row['exact_prediction_file_bytes']}" for row in scene1['feature_joint_prediction_parity'])+'. This bounded geometry attribution does not extend missing head-seed full suites or validate GT camera poses.']
    (ROOT/'docs/GEO_STREAM_DENSE_SEED_STABILITY.md').write_text('\n'.join(text)+'\n')
    print(json.dumps([dict(factor=g['factor'], method=g['method'], complete=g['completed_trials'], pending=len(g['pending'])) for g in groups]))


if __name__ == '__main__':
    main()

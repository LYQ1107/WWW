"""Import complete queued suites; combine only complete declared inputs on CPU."""
import argparse
import shutil
from common import *


def validate_report(report, source, strict):
    spec = json.loads(source.read_text())['video_predictions']
    if sorted(map(int, spec)) != strict or report['status'] != 'PASS' or not report['full_video'] or report['video_ids'] != strict:
        raise ValueError('complete declared strict suite required')
    if report['source_manifest_sha256'] != sha256(source):
        raise ValueError('suite source specification changed')
    expected = {path: sha256(path) for path in spec.values()}
    if report['prediction_sha256'] != expected:
        raise ValueError('suite evaluated different prediction files')
    if set(report['run_manifests']) != set(spec):
        raise ValueError('missing or unexpected scene manifests')
    for scene, captured in report['run_manifests'].items():
        if sha256(captured['path']) != captured['sha256']:
            raise ValueError('changed captured inference manifest')
        run = json.loads(Path(captured['path']).read_text())
        if not run['full_video'] or run['video_id'] != int(scene) or run['causal_online'] != report['causal_online']:
            raise ValueError('suite/inference scope mismatch')
        if run.get('prediction_sha256', run.get('predictions_sha256')) != expected[spec[scene]]:
            raise ValueError('inference/prediction hash mismatch')
    for path, digest in report['raw_gt_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('changed raw GT')


def ready(source, strict):
    spec = json.loads(source.read_text())['video_predictions']
    if sorted(map(int, spec)) != strict:
        raise ValueError('complete strict source specification required')
    for scene, value in spec.items():
        prediction = Path(value)
        manifest = prediction.with_name(prediction.stem+'_manifest.json')
        if not manifest.exists():
            manifest = prediction.parent/'manifest.json'
        metric = prediction.with_suffix('.metrics.json')
        if not all(p.exists() for p in (prediction, manifest, metric)):
            return False
        run, measured = [json.loads(p.read_text()) for p in (manifest, metric)]
        if not run.get('full_video') or not measured.get('full_video') or measured.get('status') != 'PASS':
            return False
        if run['video_id'] != int(scene) or measured['video_id'] != int(scene):
            raise ValueError('wrong completed input scene')
        digest = sha256(prediction)
        if digest != run.get('prediction_sha256', run.get('predictions_sha256')) or digest != measured['prediction_sha256']:
            raise ValueError('changed completed input predictions')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-ready', action='store_true', help='run native CPU combination for ready missing suites')
    args = parser.parse_args()
    plan = json.loads((ROOT/'manifests/identity_relations_matrix_v2.json').read_text())
    rng_plan = json.loads((ROOT/'manifests/primary_vggt_inference_seed_pairs_v1.json').read_text())
    # Existing TRAIN-head queues already own their combination stages. Never
    # race them for the same output directory after the final scene evaluates.
    if not rng_plan.get('combine_after_all_scenes'):
        raise ValueError('root combination ownership is not declared')
    root_combined = {f'causal_B7_primary_head_inference_seed{seed}_strict_suite_v1' for seed in rng_plan['inference_seeds']}
    strict = sorted(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    imported, pending = [], []
    for entry in plan['methods']:
        tag = entry['suite']
        if (ROOT/'results'/f'{tag}_metrics.json').exists():
            continue # existing reports are audited by audit_suite_settings/identity_matrix
        source = ROOT/'manifests'/entry['method_sources']
        locations = [RUNTIME/'matrix_reports'/f'{tag}_metrics.json', RUNTIME/'suites'/tag/'metrics.json']
        result = next((path for path in locations if path.exists()), None)
        if result is None:
            if not args.generate_ready or tag not in root_combined or not ready(source, strict):
                pending.append(tag)
                continue
            if (RUNTIME/'suites'/tag).exists():
                raise FileExistsError('incomplete suite directory retained; inspect it before any new evaluation: '+tag)
            log = RUNTIME/'logs'/f'collect_{tag}_native_combine_v1.log'
            with log.open('x') as stream:
                subprocess.run([sys.executable, str(ROOT/'geo_tools/evaluate_suite.py'), '--sources', str(source),
                    '--tag', tag, '--report-output', str(locations[0])], cwd=ROOT, check=True,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                    stdout=stream, stderr=subprocess.STDOUT)
            result = locations[0]
        report = json.loads(result.read_text())
        validate_report(report, source, strict)
        if report['causal_online'] != (entry['protocol'] == 'causal'):
            raise ValueError('declared and evaluated protocols differ')
        target = ROOT/'results'/f'{tag}_metrics.json'
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(result, target)
        imported.append(dict(suite=tag, path=str(target), sha256=sha256(target)))
    print(json.dumps(dict(imported=imported, pending=pending)))


if __name__ == '__main__':
    main()

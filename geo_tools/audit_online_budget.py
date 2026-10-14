"""Actual full-video encoder budgets and prediction parity, never inferred caps."""
import argparse
from common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    baseline = json.loads((args.baseline/'manifest.json').read_text())
    baseline_predictions = json.loads((args.baseline/'predictions.json').read_text())
    baseline_metrics = json.loads((args.baseline/'predictions.metrics.json').read_text())['metrics']
    rows = []
    for run in args.runs:
        manifest_path = run/'manifest.json'
        report = json.loads(manifest_path.read_text())
        if not report['full_video'] or not report['causal_online'] or report['video_id'] != baseline['video_id']:
            raise ValueError('same complete causal video required')
        if report['checkpoint_sha256'] != baseline['checkpoint_sha256'] or report['config_sha256'] != baseline['config_sha256']:
            raise ValueError('GMT checkpoint/config mismatch')
        predictions = json.loads((run/'predictions.json').read_text())
        if sha256(run/'predictions.json') != report['predictions_sha256']:
            raise ValueError('changed prediction artifact')
        metrics = json.loads((run/'predictions.metrics.json').read_text())['metrics']
        rows.append(dict(run=str(run), manifest_sha256=sha256(manifest_path), commit=report['commit'],
                         planned_total_cap=report['geo_options']['TOTAL_CALL_BUDGET'], actual_encoder_calls=report['total_vggt_calls'],
                         failures=report['geo_failures'], valid_geometry_calls=report['geometry_valid_calls'],
                         exact_baseline_predictions=predictions == baseline_predictions, predictions=len(predictions), metrics=metrics,
                         metric_delta={key:metrics[key]-baseline_metrics[key] for key in metrics},
                         camera_image_fps=report['camera_image_fps'], input_to_identity_p50=report['input_to_identity_p50'], input_to_identity_p95=report['input_to_identity_p95']))
    save_json(ROOT/'results'/f'{args.tag}.json', dict(status='PASS', baseline=str(args.baseline), baseline_manifest_sha256=sha256(args.baseline/'manifest.json'),
              matched_actual_calls=len({row['actual_encoder_calls'] for row in rows}) == 1,
              all_runs_without_module_failures=all(row['failures'] == 0 for row in rows), rows=rows,
              speed_scope='single descriptive measurements on actual V100 hardware; no repeated timing or hardware-normalized Pareto superiority claim'))


if __name__ == '__main__':
    main()

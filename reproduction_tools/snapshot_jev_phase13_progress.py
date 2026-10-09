"""Publish compact actual progress without uploading training/state artifacts."""
from datetime import datetime, timezone
from jev_phase13_common import *


def main():
    queues = {}
    cases = []
    for kind in ['formal', 'validation', 'onpolicy']:
        path = OUT / f'queue_{kind}_v1' / 'PROGRESS.json'
        if path.exists():
            r = json.loads(path.read_text())
            queues[kind] = {'status': r['status'], 'source_commit': r['binding']['source_commit'],
                            'done': len(r['done']), 'active': r['active'], 'pending': len(r['pending']),
                            'failed': r['failed'], 'elapsed_seconds': r['seconds'], 'path': str(path)}
        final = path.parent / 'RESULT.json'
        if final.exists():
            r = json.loads(final.read_text())
            queues[kind].update(status=r['status'], done=len(r['done']), active=[], pending=0, failed=r['failed'])
    for phase in ['tiny', 'pilot', 'formal', 'onpolicy']:
        for path in sorted((OUT / f'{phase}_training_v1').glob('*/seed*/PROGRESS.json')):
            r = json.loads(path.read_text())
            case = {'phase': phase, 'variant': path.parent.parent.name, 'seed': int(path.parent.name[4:]),
                    'progress': r, 'path': str(path)}
            final = path.parent / 'RESULT.json'
            if final.exists():
                f = json.loads(final.read_text())
                case.update(result={'path': str(final), 'SHA256': sha(final)}, checkpoint=f['checkpoint'],
                            status=f['status'], source_commit=f['binding']['source_commit'], source_root=f['binding']['source_root'])
                if phase == 'onpolicy':
                    case['paired_first256_TRAIN'] = {
                        when: {k: sum(v['summary'][k] for v in f[when + '_online_TRAIN'])
                               for k in ['birth_anchor_wrong_observations', 'extra_birth_fragments',
                                         'wrong_ID_duration_total_camera_frames']}
                        for when in ['before', 'after']}
                    case['paired_interpretation'] = 'wrong-anchor reductions must be read with extra births and full-video tracking; changing an ID resets its anchor'
            cases.append(case)
    online = {}
    for phase in ['formal', 'onpolicy']:
        root = OUT / f'{phase}_online_v1'
        complete = list(root.glob('*/video*/RESULT.json'))
        progressing = [str(p) for p in root.glob('*/video*/PROGRESS.json') if not (p.parent / 'RESULT.json').exists()]
        actual = []
        for path in sorted(complete):
            r = json.loads(path.read_text())
            actual.append({'variant': r['variant'], 'seed': r['seed'], 'case_name':path.parent.parent.name,
                           'no_calibration':path.parent.parent.name.endswith('_no_calibration'), 'video': path.parent.name,
                           'result': {'path': str(path), 'SHA256': sha(path)}, 'strict_metrics': r['strict_online_metrics'],
                           'trained': r['trained'], 'source_commit': r['binding']['source_commit'],
                           'raw_predictions': r['raw_predictions'], 'identity': r['identity_summary'],
                           'scope': 'completed individual frozen video; not pooled or seed-average evidence'})
        online[phase] = {'completed_video_runs': len(complete), 'running_progress_paths': progressing,
                         'completed_results': actual}
    pipeline=OUT/'completion_pipeline_v1/RESULT.json'
    completion={'status':json.loads(pipeline.read_text())['status'],'result':{'path':str(pipeline),'SHA256':sha(pipeline)}} if pipeline.exists() else {'status':'IN_PROGRESS'}
    live=list((OUT/'live_efficiency_v1').glob('*/seed*/RESULT.json'))
    save(REPORTS / 'RUN_PROGRESS.json', {'status': 'COMPLETE' if completion['status']=='PASS' else 'IN_PROGRESS', 'UTC': datetime.now(timezone.utc).isoformat(),
                                        'binding': binding(), 'queues': queues, 'training_cases': cases, 'online': online,
                                        'completion':completion, 'completed_real_live_trials':len(live),
                                        'tracking_scope': 'final pooled comparisons and scientific gate outcomes are in FINAL_GO_NO_GO.json; completion does not imply scientific GO',
                                        'large_checkpoints_and_state_logs_uploaded': False, 'heldout': 'SEALED'})
    print('PHASE13_PROGRESS_CAPTURED', [(k, v['done'], len(v['active']), v['pending']) for k, v in queues.items()], flush=True)


if __name__ == '__main__':
    main()

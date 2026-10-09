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
                            source_commit=f['binding']['source_commit'], source_root=f['binding']['source_root'])
            cases.append(case)
    online = {}
    for phase in ['formal', 'onpolicy']:
        root = OUT / f'{phase}_online_v1'
        complete = list(root.glob('*/video*/RESULT.json'))
        progressing = [str(p) for p in root.glob('*/video*/PROGRESS.json') if not (p.parent / 'RESULT.json').exists()]
        online[phase] = {'completed_video_runs': len(complete), 'running_progress_paths': progressing}
    save(REPORTS / 'RUN_PROGRESS.json', {'status': 'IN_PROGRESS', 'UTC': datetime.now(timezone.utc).isoformat(),
                                        'binding': binding(), 'queues': queues, 'training_cases': cases, 'online': online,
                                        'tracking_scope': 'partial optimizer/audit progress only; final pooled HOTA/AssA conclusions require completed frozen full-video runs',
                                        'large_checkpoints_and_state_logs_uploaded': False, 'heldout': 'SEALED'})
    print('PHASE13_PROGRESS_CAPTURED', [(k, v['done'], len(v['active']), v['pending']) for k, v in queues.items()], flush=True)


if __name__ == '__main__':
    main()

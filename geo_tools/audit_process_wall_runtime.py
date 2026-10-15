"""Observed launch-to-exit costs, excluding recorded held-dispatcher stages."""
import csv
from common import *


ENTRIES = {'run_online.py', 'run_seeded_online.py', 'run_native_prior.py', 'run_live_retrieval.py'}


def record_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def process_cost(stage, manifest, held):
    if held:
        return None # parent SIGSTOP can extend child.wait() after the child exits
    if stage['status'] != 'PASS' or stage['exit_code'] != 0:
        raise ValueError('completed successful process stage required')
    if not manifest['full_video'] or not manifest.get('speed_eligible', True) or manifest.get('used_cache') is not False:
        raise ValueError('complete uncached production replay required')
    outer, loop = stage['seconds'], manifest['seconds']
    if not 0 < loop <= outer or manifest['images'] <= 0:
        raise ValueError('inconsistent measured process/loop durations')
    return dict(process_wall_seconds=outer, reported_loop_seconds=loop, outside_loop_seconds=outer-loop,
                process_camera_image_fps=manifest['images']/outer, reported_loop_camera_image_fps=manifest['images']/loop)


def affected_windows(controller_pid, stage, windows):
    def seconds(value):
        return datetime.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    start, end = seconds(stage['started_utc']), seconds(stage['finished_utc'])
    found = []
    for window in windows:
        if controller_pid != window['controller_pid']:
            continue
        # Direct child records also cover the wait before the priority queue
        # starts. Conservative lifetime bounds cover suspension just before
        # Popen, when the dispatcher's /proc children list can still be empty.
        child = str(stage['pid']) in window['existing_children']
        overlap = start <= window['conservative_end_timestamp'] and end >= window['conservative_start_timestamp']
        if child or overlap:
            found.append(window['path'])
    return found


def main():
    windows = []
    for path in sorted((RUNTIME/'priority_windows').glob('*/manifest.json')):
        info = json.loads(path.read_text())
        completed = info['status'] in {'PRIORITY_COMMANDS_ENDED', 'INTERRUPTED'}
        windows.append(dict(path=str(path), observed_status=info['status'], controller_pid=info['controller_pid'],
            controller_starttime=info['controller_starttime'], existing_children=info.get('existing_children', {}),
            conservative_start_timestamp=datetime.datetime.fromisoformat(info['provenance']['process_started_utc']).timestamp(),
            conservative_end_timestamp=path.stat().st_mtime if completed else datetime.datetime.now(datetime.timezone.utc).timestamp(),
            bound_scope='priority owner import time through final report mtime, or observation time while live; can conservatively exclude additional waiting-predecessor stages'))
    rows, excluded, stages, artifacts = [], [], {}, {}
    for queue in sorted((RUNTIME/'queues').glob('*/manifest.json')):
        info = json.loads(queue.read_text())
        for stage in info.get('stages', []):
            command = stage.get('command', [])
            entry = next((Path(x).name for x in command if Path(x).name in ENTRIES), None)
            if stage.get('status') != 'PASS' or entry is None or '--tag' not in command:
                continue
            tag = command[command.index('--tag')+1]
            family = 'retrieval' if entry == 'run_live_retrieval.py' else 'online'
            path = RUNTIME/family/tag/'manifest.json'
            if not path.exists():
                raise FileNotFoundError(path)
            run = json.loads(path.read_text())
            if not run['full_video'] or not run.get('speed_eligible', True):
                continue
            key = str(queue)+'::'+stage['id']
            stages[key] = dict(queue=str(queue), controller_pid=info['pid'], stage=stage,
                               stage_record_sha256=record_digest(stage))
            affected = affected_windows(info['pid'], stage, windows)
            if affected:
                excluded.append(dict(method=tag, stage_key=key, priority_windows=affected,
                    reason='recorded held child or overlap with conservative priority-owner lifetime; queue-observed duration may include suspension'))
                continue
            if sha256(stage['log']) != stage['log_sha256']:
                raise ValueError('completed process log changed')
            if run.get('raw_log_path') != stage['log']:
                raise ValueError('inference and queue process log differ')
            cost = process_cost(stage, run, False)
            if family == 'online':
                prediction = path.parent/'predictions.json'
                if sha256(prediction) != run['predictions_sha256']:
                    raise ValueError('changed process prediction')
            else:
                for filename, digest in run['artifacts'].items():
                    if sha256(path.parent/filename) != digest:
                        raise ValueError('changed retrieval process artifact')
            rows.append(dict(method=tag, family=family, images=run['images'], video_id=run.get('video_id'),
                manifest=str(path), manifest_sha256=sha256(path), stage_key=key, **cost))
            artifacts[str(path)] = sha256(path)
            artifacts[stage['log']] = stage['log_sha256']
    index = {row['manifest']: row for row in rows}
    if len(index) != len(rows):
        raise ValueError('duplicate completed process for the same inference artifact')
    strict = sorted(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    suites = []
    for path in sorted((ROOT/'results').glob('*_metrics.json')):
        measured = json.loads(path.read_text())
        captured = measured.get('run_manifests', {})
        if measured.get('status') != 'PASS' or not measured.get('causal_online') or not measured.get('full_video') or measured.get('video_ids') != strict:
            continue
        if len(captured) != len(strict) or any(v['path'] not in index for v in captured.values()):
            continue # no manufactured setup estimate for missing or held stages
        selected = [index[captured[str(scene)]['path']] for scene in strict]
        if any(captured[str(scene)]['sha256'] != row['manifest_sha256'] for scene, row in zip(strict, selected)):
            raise ValueError('suite and process captures differ')
        images = sum(row['images'] for row in selected)
        outer = sum(row['process_wall_seconds'] for row in selected)
        loop = sum(row['reported_loop_seconds'] for row in selected)
        suites.append(dict(suite=path.stem.removesuffix('_metrics'), images=images, full_strict_suite=True,
            process_wall_seconds=outer, reported_loop_seconds=loop, outside_loop_seconds=outer-loop,
            process_camera_image_fps=images/outer, reported_loop_camera_image_fps=images/loop,
            source=str(path), source_sha256=sha256(path), stage_keys=[row['stage_key'] for row in selected]))
        artifacts[str(path)] = sha256(path)
    result = dict(status='RECORDED_UNHELD_PROCESS_WALL_COSTS', rows=rows, suites=suites, excluded_held_stages=excluded,
        stage_snapshots=stages, observed_priority_windows=windows, source_artifacts=artifacts,
        scope='actual successful full replay stages: parent monotonic time immediately around launch/wait, including Python startup, model setup, replay and export; excludes inter-stage queue waits and separate GT evaluation',
        limitations='documented held children and conservative priority-owner lifetime overlaps excluded, even if their loop timing is valid; this can also exclude extra predecessor-wait stages; stage snapshots are immutable completed records within mutable queue manifests; outside-loop difference includes imports, setup, export/provenance and normal host scheduling, not pure model setup; cumulative suite FPS is sum(images)/sum(times), not mean FPS, one descriptive run per input, no stable speed or camera/network capture-latency claim', provenance=provenance())
    save_json(ROOT/'results/process_wall_runtime_v1.json', result)
    with (ROOT/'results/process_wall_runtime_v1.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    text = ['# Observed complete process replay cost', '', result['scope']+'.', '', result['limitations']+'.', '',
        '| Complete causal strict5 suite | Images | Reported loop images/s | Observed process images/s | Outside-loop seconds |',
        '|---|---:|---:|---:|---:|']
    for row in suites:
        text.append(f"| {row['suite']} | {row['images']} | {row['reported_loop_camera_image_fps']:.6f} | {row['process_camera_image_fps']:.6f} | {row['outside_loop_seconds']:.3f} |")
    text += ['', f"Recorded full process rows: {len(rows)}; excluded full held-dispatcher stages: {len(excluded)}. Missing complete suites are not filled with zero setup cost. Sealed stage/log/manifest hashes and separate full retrieval rows remain in the JSON/CSV."]
    (ROOT/'docs/GEO_STREAM_PROCESS_WALL_RUNTIME.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(processes=len(rows), complete_suites=len(suites), excluded_held=len(excluded))))


if __name__ == '__main__':
    main()

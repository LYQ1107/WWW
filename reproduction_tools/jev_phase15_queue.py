"""Lightweight immutable actors and GPU scheduling with readable progress."""
import argparse
import time
from jev_phase15_common import *


def pin(name):
    protect(); storage_guard(); OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / name
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if destination.exists():
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=destination, text=True).strip() == head
        return destination
    subprocess.run(['git', 'worktree', 'add', '--no-checkout', '--detach', str(destination), head], cwd=ROOT, check=True)
    subprocess.run(['git', 'read-tree', 'HEAD'], cwd=destination, check=True)
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=destination).split(b'\0'); keep = []; skip = []
    for rel in filter(None, paths):
        text = rel.decode()
        omit = (text.startswith('reports/') and not text.startswith(('reports/JEV_PHASE13/', 'reports/JEV_PHASE14/', 'reports/JEV_PHASE15/'))) or text.startswith(('TrackEval/data/', '"TrackEval/'))
        (skip if omit else keep).append(rel)
    subprocess.run(['git', 'update-index', '--skip-worktree', '-z', '--stdin'], input=b'\0'.join(skip) + b'\0', cwd=destination, check=True)
    subprocess.run(['git', 'checkout-index', '-z', '--stdin'], input=b'\0'.join(keep) + b'\0', cwd=destination, check=True)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=destination, text=True).strip()
    save(OUT / (name + '.json'), {'status': 'PINNED', 'path': str(destination), 'commit': head,
         'checked_out_files': len(keep), 'skipped_irrelevant_files': len(skip), 'historical_tree_unchanged': True})
    return destination


def gpus():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    return [tuple(map(int, row.split(','))) for row in rows.strip().splitlines()]


def run_queue(source, jobs, name, max_active=4):
    source = Path(source); head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()
    folder = OUT / name; folder.mkdir(parents=True, exist_ok=True)
    pending = list(jobs); active = {}; done = []; failed = []; launches = []; begin = time.monotonic()
    while pending or active:
        for gpu, (proc, log, job) in list(active.items()):
            code = proc.poll()
            if code is None: continue
            log.close(); item = dict(job, returncode=code)
            good = code == 0 and Path(job['result']).exists()
            (done if good else failed).append(item); del active[gpu]
            print('PHASE15_JOB_FINISHED', job['key'], code, flush=True)
        candidates = [g for g in gpus() if g[0] not in active and g[2] >= 8192]
        candidates.sort(key=lambda g: (g[1] >= 1000, g[3], g[1], g[0]))
        for gpu, used, free, utilization in candidates:
            if not pending or len(active) >= max_active: break
            storage_guard(); job = pending.pop(0)
            if Path(job['result']).exists():
                result = read(job['result'])
                assert result['binding']['source_commit'] == head
                done.append(dict(job, returncode=0, resumed_completed=True)); continue
            env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            path = folder / (job['key'] + '.log'); log = path.open('a')
            command = [PYTHON, str(source / 'reproduction_tools' / job['script'])] + job['args']
            proc = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = proc, log, job
            launches.append({'key': job['key'], 'pid': proc.pid, 'GPU': gpu, 'command': command,
                             'source_commit': head, 'log': str(path), 'GPU_free_MiB_at_launch': free})
            print('PHASE15_JOB_LAUNCH', gpu, proc.pid, job['key'], flush=True)
        save(folder / 'PROGRESS.json', {'status': 'RUNNING', 'total': len(jobs), 'done': len(done),
            'failed': len(failed), 'pending': len(pending), 'seconds': time.monotonic() - begin,
            'active': [{'GPU': gpu, 'pid': proc.pid, 'key': job['key'], 'result': job['result']} for gpu, (proc, _, job) in active.items()]})
        if pending or active: time.sleep(10)
    result = {'status': 'COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES',
              'binding': binding(), 'actor_source_commit': head, 'done': done, 'failed': failed,
              'launches': launches, 'seconds': time.monotonic() - begin}
    save(folder / 'RESULT.json', result); return result


def main():
    source = pin('source_replay_v1'); jobs = []
    pairs = [('multi_question', 20261009), ('fixed_question', 20261009), ('set_transformer', 20261009),
             ('multi_question', 20261008), ('multi_question', 20261010)]
    for variant, seed in pairs:
        for video in DEV:
            jobs.append({'key': f'{variant}_seed{seed}_video{video}', 'script': 'jev_phase15_native_replay.py',
                         'args': ['--variant', variant, '--seed', str(seed), '--video', str(video)],
                         'result': str(OUT / 'native_replay_v1' / f'{variant}_seed{seed}' / f'video{video:02d}' / 'RESULT.json')})
    result = run_queue(source, jobs, 'native_replay_queue_v1')
    assert not result['failed'], result['failed']


if __name__ == '__main__':
    main()

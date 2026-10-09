"""Wait for owned queues, then execute all remaining frozen evaluations.

No model/config selection, Git writes, heldout access, or synthetic results.
Reporting subprocesses run in the development checkout; actual actors keep
using this clean immutable source pin.
"""
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from jev_phase13_learning import *

DEV = Path('/home/liuyeqiang/WWW_jev_phase13')


def run(script, arguments, logfile, gpu=None, root=ROOT):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    with Path(logfile).open('a') as out:
        subprocess.run([PYTHON, str(root / 'reproduction_tools' / script)] + list(arguments),
                       cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)


def uncalibrated(seed, gpu):
    for video in VAL:
        path = OUT / 'formal_online_v1' / f'full_seed{seed}_no_calibration' / f'video{video:02d}' / 'RESULT.json'
        if path.exists():
            assert json.loads(path.read_text())['status'] == 'COMPLETE'
            continue
        run('evaluate_jev_stage2_online.py', ['--variant', 'full', '--seed', str(seed), '--video', str(video), '--no-calibration'],
            OUT / f'no_calibration_seed{seed}.log', gpu)


def main():
    protect()
    source = binding()
    assert not source['dirty']
    root = OUT / 'completion_pipeline_v1'
    root.mkdir(parents=True, exist_ok=True)
    for kind in ['formal', 'validation', 'onpolicy']:
        while not (OUT / f'queue_{kind}_v1' / 'RESULT.json').exists():
            save(root / 'PROGRESS.json', {'status': 'WAITING', 'stage': kind, 'binding': source})
            time.sleep(15)
        result = json.loads((OUT / f'queue_{kind}_v1' / 'RESULT.json').read_text())
        assert result['status'] == 'PASS', (kind, result['failed'])
        print('PHASE13_DEPENDENCY_COMPLETE', kind, flush=True)
        if kind == 'formal':
            run('finalize_jev_phase13_training.py', ['--phase', 'formal'], root / 'supervised_aggregate.log', root=DEV)
    save(root / 'PROGRESS.json', {'status': 'RUNNING', 'stage': 'actual_no_calibration', 'binding': source})
    # Queues are finished: three independent empty V100s, one worker/card.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(uncalibrated, seed, gpu) for seed, gpu in zip(SEEDS, [0, 2, 3])]
        for future in futures:
            future.result()
    print('PHASE13_NO_CALIBRATION_COMPLETE', flush=True)
    from compare_jev_stage2_baselines import pooled
    for seed in SEEDS:
        pooled('full', seed, no_calibration=True)
    # Serial trials on one empty GPU avoid sharing benchmark compute with
    # our own training/inference and make complete image I/O timing explicit.
    for variant in ['full', 'cosine'] + [v for v in VARIANTS if v != 'full']:
        save(root / 'PROGRESS.json', {'status': 'RUNNING', 'stage': 'real_live_efficiency', 'variant': variant, 'binding': source})
        path = OUT / 'live_efficiency_v1' / variant / 'seed20261009' / 'RESULT.json'
        if not path.exists():
            run('benchmark_jev_phase13_live.py', ['--variant', variant, '--seed', '20261009'], root / f'live_{variant}.log', 0)
        assert json.loads(path.read_text())['status'] == 'COMPLETE'
        print('PHASE13_LIVE_TRIAL_COMPLETE', variant, flush=True)
    for phase in ['formal', 'onpolicy']:
        save(root / 'PROGRESS.json', {'status': 'RUNNING', 'stage': 'actual_pooled_TrackEval', 'phase': phase, 'binding': source})
        run('compare_jev_stage2_baselines.py', ['--phase', phase, '--only', 'all'], root / f'{phase}_pooled.log', root=DEV)
    run('finalize_jev_phase13_results.py', [], root / 'final_report.log', root=DEV)
    save(root / 'RESULT.json', {'status': 'PASS', 'binding': source, 'final_report': str(DEV / 'reports/JEV_PHASE13/FINAL_GO_NO_GO.json'),
                               'scope': 'required execution complete; scientific gates may honestly FAIL/NO_GO and lifecycle qualification may remain NOT_RUN'})
    save(root / 'PROGRESS.json', {'status': 'COMPLETE', 'stage': 'reports', 'binding': source})
    print('PHASE13_COMPLETION_PIPELINE_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        save(OUT / 'completion_pipeline_v1/FAILED.json', {'status': 'FAIL', 'binding': binding(), 'traceback': traceback.format_exc()})
        raise

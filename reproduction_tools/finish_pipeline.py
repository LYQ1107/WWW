"""Wait for the current training controller, then run verified inference/evaluation once."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
lock = (ROOT / 'manifests/finish_pipeline.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state = {'pid': os.getpid(), 'jobs': []}


def save(status, **extra):
    state.update(status=status, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    (ROOT / 'manifests/finish_pipeline_state.json').write_text(json.dumps(state, indent=2))
    print(status, extra, flush=True)


def run(label, args):
    with (ROOT / 'reports/commands_used.sh').open('a') as commands:
        commands.write('\n# ' + label + '\n' + shlex.join(args) + '\n')
    with (ROOT / 'logs' / (label + '_controller.log')).open('xb') as log:
        process = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        job = {'label': label, 'pid': process.pid, 'args': args}
        state['jobs'].append(job)
        save(label)
        job['exit_code'] = process.wait()
        save(label + '_exited')
        if job['exit_code']:
            raise RuntimeError(label + ' failed; preserve artifacts and inspect logs before retry')


def training_state():
    # The existing controller writes this small JSON in place.
    for attempt in range(5):
        try:
            return json.loads((ROOT / 'manifests/training_pipeline_state.json').read_text())
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(0.2)


try:
    initial = training_state()
    controller = initial['pid']
    proc = Path('/proc') / str(controller)
    assert proc.stat().st_uid == os.getuid()
    assert str(ROOT / 'tools/train_pipeline.py').encode() in (proc / 'cmdline').read_bytes()
    identity = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    save('waiting_for_training', training_controller_pid=controller)
    while True:
        training = training_state()
        if training['pid'] != controller or training['current'].startswith('FAILED'):
            raise RuntimeError('Training controller changed or reported failure')
        if training['current'] == 'Training stages complete; inference/evaluation preparation required':
            assert all((ROOT / f'manifests/stage{s}_complete.json').is_file() for s in (1, 2))
            break
        if not proc.exists():
            raise RuntimeError('Training controller exited before both stages were verified')
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z' or fields[19] != identity:
            raise RuntimeError('Training controller no longer live with original process identity')
        time.sleep(30)
    python = str(ROOT / 'tools/miniconda3/envs/GMT/bin/python')
    run('trained_inference', [python, str(ROOT / 'tools/run_trained_inference.py')])
    run('prediction_conversion', [python, str(ROOT / 'tools/prepare_prediction_evaluation.py'),
        '--predictions', str(ROOT / 'outputs/predictions_stage2'), '--output', str(ROOT / 'outputs/evaluation_inputs')])
    run('real_evaluation', [python, str(ROOT / 'tools/run_real_evaluation.py')])
    save('real_evaluation_finished_pending_final_audit')
except Exception as error:
    save('FAILED_no_automatic_retry', error=str(error))
    raise

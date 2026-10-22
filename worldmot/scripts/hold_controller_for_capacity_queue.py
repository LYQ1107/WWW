#!/usr/bin/env python3
"""Hold only the registered controller after four final control fits start.

The fits continue. At their completed-checkpoint boundary the agent can reload
the optional capacity-study queue without changing running training source.
"""
import json
import os
import signal
import time
from pathlib import Path

from cache_fulltrain import atomic_json


ROOT = Path('/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022')
CONTROLLER = 12419
CONTROLS = ('no_omega', 'no_appearance', 'two_stage', 'no_geometry')
EXPECTED_CWD = Path(__file__).resolve().parents[1]


def active_controls():
    found = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if b'scripts/train_world_query.py' not in args or (proc / 'cwd').resolve() != EXPECTED_CWD:
                continue
            for method in CONTROLS:
                checkpoint = str(ROOT / 'final' / (method + '_seed20261022.pt')).encode()
                if checkpoint in args:
                    found[method] = int(proc.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


if __name__ == '__main__':
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        fits = active_controls()
        if len(fits) == 4:
            proc = Path('/proc') / str(CONTROLLER)
            if (b'scripts/orchestrate_fulltrain_test.py' not in (proc / 'cmdline').read_bytes().split(b'\0')
                    or (proc / 'cwd').resolve() != EXPECTED_CWD):
                raise RuntimeError('controller PID or working directory changed')
            os.kill(CONTROLLER, signal.SIGSTOP)
            record = {'status': 'CONTROLLER_ONLY_WAITING_FOR_COMPLETE_CONTROL_FITS',
                      'controller_pid': CONTROLLER, 'live_control_fits': fits,
                      'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'training_children_signalled': False,
                      'reason': 'reload separately registered capacity research as a fourth-card queue job after current training',
                      'next_step': 'after all seven final fits complete, replace only this stopped controller; preserve all checkpoints and selected settings'}
            atomic_json(ROOT / 'capacity_queue_boundary_hold.json', record)
            print(json.dumps(record), flush=True)
            break
        time.sleep(5)
    else:
        raise RuntimeError('control-fit start boundary was not observed; inspect controller before any signal')

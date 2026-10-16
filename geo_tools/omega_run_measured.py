"""Charge the complete child process, including imports/setup and final export."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from omega_common import ROOT, provenance, save_new, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.receipt.exists() or args.log.exists():
        raise ValueError('new receipt/log and explicit command required')
    if args.command[0] == '--':
        args.command = args.command[1:]
    source = provenance()
    if source['source_status'].strip():
        raise ValueError('commit source before launching a measured experiment')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.log.open('x') as log:
        child = subprocess.Popen(args.command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        code = child.wait()
    elapsed = time.monotonic()-started
    receipt = dict(status='CHILD_FINISHED' if code == 0 else 'CHILD_FAILED', command=args.command,
                   exit_code=code, pid=child.pid, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   child_process_wall_seconds=elapsed, GPU_visible_process_wall_hours=elapsed/3600 if os.environ.get('CUDA_VISIBLE_DEVICES') else None,
                   timing_scope='complete child imports, input/weight reads, source/model construction, preprocessing, forwards, background checks, final export; controller provenance/log hashing excluded',
                   log=dict(path=str(args.log), sha256=sha256(args.log)), provenance=source)
    save_new(args.receipt, receipt)
    print(receipt['status'], code, elapsed, flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()

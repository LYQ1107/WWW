"""Official training entry with CPU metadata hooks and parent-owned native tracing."""
import os
from pathlib import Path
import threading
import diagnose_native_backward as observation


def worker(args, parent, output):
    observation.OUT = Path(output)
    # None preserves official setup's configured/generated seed on each rank.
    return observation.worker(args, None, parent)


if __name__ == '__main__':
    args = observation.default_argument_parser().parse_args()
    positions = [i for i, value in enumerate(args.opts) if value == 'OUTPUT_DIR']
    if len(positions) != 1:
        raise RuntimeError('Observed launcher requires one explicit OUTPUT_DIR override')
    output = Path(args.opts[positions[0] + 1]).resolve()
    observation.OUT = output
    args.dist_url = 'tcp://127.0.0.1:{}'.format(observation.torch.randint(11111, 60000, (1,))[0].item())
    args.eval_only = False
    print('OBSERVED OFFICIAL TRAINING; no seed, iteration, optimizer or loss override added', args, flush=True)
    done = threading.Event()
    monitor = threading.Thread(target=observation.monitor, args=(done,), daemon=True)
    monitor.start()
    try:
        observation.launch(worker, args.num_gpus, num_machines=args.num_machines,
            machine_rank=args.machine_rank, dist_url=args.dist_url,
            args=(args, os.getpid(), str(output)))
    finally:
        done.set()
        monitor.join(timeout=50)

"""Passive device/process samples for a finite set of existing project jobs."""
import argparse
import csv
import io
import time
from common import *
from run_research_queue import process_identity


def descendants(roots):
    found = set()
    todo = list(roots)
    while todo:
        pid = todo.pop()
        if pid in found:
            continue
        state = process_identity(pid)
        if state is None:
            continue
        found.add(pid)
        path = Path(f'/proc/{pid}/task/{pid}/children')
        try:
            todo.extend(map(int, path.read_text().split()))
        except FileNotFoundError:
            pass
    return found


def query(arguments):
    value = subprocess.check_output(['nvidia-smi', *arguments, '--format=csv,noheader,nounits'], text=True)
    return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(value))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--controller-pids', type=int, nargs='+', required=True)
    parser.add_argument('--interval', type=float, default=2.)
    args = parser.parse_args()
    if args.interval < 1:
        raise ValueError('sample interval must be at least one second')
    roots = {}
    for pid in args.controller_pids:
        state = process_identity(pid)
        if state is None or not any(name in state[1] for name in ('geo_tools/run_research_queue.py', 'geo_tools/run_priority_window.py')):
            raise ValueError('not a live named project controller: '+str(pid))
        roots[pid] = state
    out = RUNTIME/'resource_samples'/args.tag
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    report = dict(status='SAMPLING', pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        gpu_ids=[0, 4, 5, 6], interval_seconds=args.interval,
        controllers={str(pid): dict(start_ticks=state[0], command=state[1]) for pid, state in roots.items()},
        scope='passive NVIDIA device samples and live named-controller descendants, CPU RSS per PID; no inference code/weights/outputs changed',
        limitations='device utilization is a sampled whole-device measure, not per-kernel or uniquely attributable to one method; sampled maxima are not exact lifetime peaks; RSS is not unique physical memory and must not be summed as physical RAM; already-running jobs have partial coverage; no capture/network latency', provenance=provenance())
    save_json(out/'manifest.json', report)
    count = 0
    seen = {}
    try:
        with (out/'devices.csv').open('x', newline='') as df, (out/'processes.csv').open('x', newline='') as pf:
            devices = csv.writer(df, lineterminator='\n')
            processes = csv.writer(pf, lineterminator='\n')
            devices.writerow(['utc', 'monotonic_seconds', 'gpu', 'uuid', 'gpu_utilization_percent', 'memory_utilization_percent', 'device_memory_mib', 'unattributed_compute_pids'])
            processes.writerow(['utc', 'monotonic_seconds', 'pid', 'start_ticks', 'gpu', 'nvml_memory_mib', 'host_rss_kib', 'tag', 'role'])
            while True:
                live = [pid for pid, state in roots.items() if (current := process_identity(pid)) is not None and current[0] == state[0]]
                retained = [pid for pid, ticks in seen.items() if (current := process_identity(pid)) is not None and current[0] == ticks]
                owned = descendants([*live, *retained])
                if not owned:
                    break
                utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
                stamp = time.monotonic()
                gpu_rows = query(['-i', '0,4,5,6', '--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used'])
                by_uuid = {row[1]: row[0] for row in gpu_rows}
                apps = [row for row in query(['--query-compute-apps=gpu_uuid,pid,used_memory']) if row[0] in by_uuid]
                memory = {(int(row[1]), by_uuid[row[0]]): row[2] for row in apps}
                for gpu, uuid, util, mem_util, used in gpu_rows:
                    other = sum(int(row[1]) not in owned for row in apps if row[0] == uuid)
                    devices.writerow([utc, stamp, gpu, uuid, util, mem_util, used, other])
                for pid in sorted(owned):
                    state = process_identity(pid)
                    if state is None:
                        continue
                    seen[pid] = state[0]
                    try:
                        status = Path(f'/proc/{pid}/status').read_text().splitlines()
                        argv = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
                    except FileNotFoundError:
                        continue
                    rss = next((line.split()[1] for line in status if line.startswith('VmRSS:')), '')
                    tag = argv[argv.index('--tag')+1] if '--tag' in argv else ''
                    role = next((Path(value).name for value in argv if value.endswith('.py')), 'control_or_worker')
                    allocated = [(gpu, value) for (process, gpu), value in memory.items() if process == pid]
                    for gpu, used in allocated or [('', '')]:
                        processes.writerow([utc, stamp, pid, state[0], gpu, used, rss, tag, role])
                df.flush()
                pf.flush()
                count += 1
                report.update(samples=count, last_sample_utc=utc)
                save_json(out/'manifest.json', report)
                time.sleep(args.interval)
        report['status'] = 'FINISHED_EXISTING_JOBS'
    except BaseException as error:
        report.update(status='FAILED', error=repr(error))
        raise
    finally:
        report.update(samples=count, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      artifacts={p.name: sha256(p) for p in out.glob('*.csv')})
        save_json(out/'manifest.json', report)


if __name__ == '__main__':
    main()

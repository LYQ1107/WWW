"""Summarize a finished passive probe; sampled extrema are not lifetime peaks."""
import argparse
import csv
import math
from collections import defaultdict
from common import *


def number(value):
    if value in ('', 'N/A', '[N/A]'):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('invalid resource measurement')
    return result


def summarize(manifest, devices, processes):
    if manifest['status'] != 'FINISHED_EXISTING_JOBS':
        raise ValueError('final aggregation requires a successfully finished probe')
    expected = set(map(str, manifest['gpu_ids']))
    groups, by_gpu, by_process = defaultdict(dict), defaultdict(list), defaultdict(list)
    for row in devices:
        stamp = float(row['monotonic_seconds'])
        gpu = row['gpu']
        if gpu not in expected or gpu in groups[stamp]:
            raise ValueError('unexpected or duplicated device sample')
        groups[stamp][gpu] = row
        by_gpu[gpu].append(row)
    if len(groups) != manifest['samples'] or any(set(group) != expected for group in groups.values()):
        raise ValueError('missing device sample/group')
    times = sorted(groups)
    if not times:
        raise ValueError('empty probe')
    for row in processes:
        stamp = float(row['monotonic_seconds'])
        if stamp not in groups or row['utc'] != next(iter(groups[stamp].values()))['utc']:
            raise ValueError('process sample has no matching device sample')
        if row['gpu'] and row['gpu'] not in expected:
            raise ValueError('process allocated outside probe devices')
        by_process[(row['pid'], row['start_ticks'], row['tag'], row['role'])].append(row)
    def maximum(rows, key):
        values = [value for row in rows if (value := number(row[key])) is not None]
        return max(values) if values else None
    device_rows = []
    for gpu, rows in sorted(by_gpu.items()):
        stamps = [float(row['monotonic_seconds']) for row in rows]
        if stamps != sorted(set(stamps)):
            raise ValueError('device sample time is not strictly increasing')
        values = [value for row in rows if (value := number(row['gpu_utilization_percent'])) is not None]
        device_rows.append(dict(gpu=int(gpu), samples=len(rows),
            sampled_gpu_utilization_max_percent=maximum(rows, 'gpu_utilization_percent'),
            gpu_utilization_unweighted_sample_mean_percent=sum(values)/len(values) if values else None,
            sampled_device_memory_max_mib=maximum(rows, 'device_memory_mib'),
            samples_with_unattributed_compute=sum(number(row['unattributed_compute_pids']) > 0 for row in rows)))
    process_rows = []
    for (pid, ticks, tag, role), rows in sorted(by_process.items()):
        stamps = sorted({float(row['monotonic_seconds']) for row in rows})
        per_gpu = defaultdict(list)
        for row in rows:
            if row['gpu']:
                per_gpu[row['gpu']].append(row)
        process_rows.append(dict(pid=int(pid), start_ticks=ticks, tag=tag, role=role,
            samples=len(stamps), observed_span_seconds=stamps[-1]-stamps[0],
            first_sample_utc=min(row['utc'] for row in rows), last_sample_utc=max(row['utc'] for row in rows),
            present_at_first_probe_sample=stamps[0] == times[0],
            sampled_host_rss_max_kib=maximum(rows, 'host_rss_kib'),
            sampled_nvml_memory_max_mib_by_gpu={gpu: maximum(values, 'nvml_memory_mib') for gpu, values in per_gpu.items()}))
    gaps = [right-left for left, right in zip(times, times[1:])]
    return dict(device_rows=device_rows, process_rows=process_rows, sample_groups=len(times),
        observed_span_seconds=times[-1]-times[0], maximum_sample_gap_seconds=max(gaps) if gaps else None,
        requested_interval_seconds=manifest['interval_seconds'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=Path, default=RUNTIME/'resource_samples/remaining_declared_jobs_v1')
    parser.add_argument('--report-output', type=Path, default=ROOT/'results/remaining_job_sampled_resources_v1.json')
    args = parser.parse_args()
    path = args.samples/'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['status'] != 'FINISHED_EXISTING_JOBS':
        raise ValueError('probe is not finished; no final resource report written')
    if args.report_output.exists() and str(path) not in json.loads(args.report_output.read_text())['source_artifacts']:
        raise FileExistsError('different probe requires a separate report output')
    artifacts = {str(path): sha256(path)}
    data = {}
    for name in ('devices.csv', 'processes.csv'):
        source = args.samples/name
        if sha256(source) != manifest['artifacts'][name]:
            raise ValueError('changed finished probe CSV')
        artifacts[str(source)] = sha256(source)
        with source.open() as stream:
            data[name] = list(csv.DictReader(stream))
    result = summarize(manifest, data['devices.csv'], data['processes.csv'])
    result.update(status='FINISHED_PROBE_SAMPLED_RESOURCES', source_artifacts=artifacts,
        scope='passive whole-device and per-process samples for the named existing job controllers and descendants; includes controls/evaluation/workers separately',
        limitations='sampled extrema are not exact lifetime peaks; present at first probe sample indicates possible partial coverage, while later appearance does not prove complete lifetime coverage; sample means are unweighted, not time-weighted utilization; no unique per-method device attribution; per-PID RSS maxima are never summed as physical RAM, and per-device/per-process noncoincident maxima are never summed; overlapping probes are not independent repetitions and must not be summed or interpreted as uniquely attributable user workloads; no throughput or capture/network-latency estimate',
        provenance=provenance())
    save_json(args.report_output, result)
    print(json.dumps(dict(status=result['status'], sample_groups=result['sample_groups'], processes=len(result['process_rows']))))


if __name__ == '__main__':
    main()

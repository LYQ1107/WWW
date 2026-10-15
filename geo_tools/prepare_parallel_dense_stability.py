"""Repartition unchanged, unstarted dense stability suites across allocated GPUs."""
import re
from common import *


def main():
    source=ROOT/'manifests/dense_stability_and_camera_queue_v1.json'
    original=json.loads(source.read_text())['queues']['gpu5']
    queues={f'gpu{gpu}':[] for gpu in (0,4,5,6)}
    assignment={('5','20261014'):'gpu5',('5','20261015'):'gpu5',('5','20261016'):'gpu4',
                ('7','20261015'):'gpu0',('7','20261016'):'gpu6'}
    for stage in original:
        match=re.search(r'causal_B([57])_dense_seed(2026101[456])_',stage['id'])
        if match is None:raise ValueError('unrecognized stability stage')
        queues[assignment[match.groups()]].append(stage)
    flattened=[stage for stages in queues.values() for stage in stages]
    if {s['id']:s for s in flattened}!={s['id']:s for s in original} or len(flattened)!=len(original):
        raise ValueError('repartition changed, duplicated or omitted experiments')
    for stages in queues.values():
        seen=set()
        for stage in stages:
            if not set(stage['depends_on'])<=seen:raise ValueError('dependency split across queues')
            seen.add(stage['id'])
    save_json(ROOT/'manifests/dense_stability_parallel_v2.json',dict(id='dense_stability_parallel_v2',
        project_gpu_ids=['0','4','5','6'],queues=queues,unchanged_source_plan=str(source),unchanged_source_plan_sha256=sha256(source),
        execution_source_commit='ac5d940e',
        scope='same53 original stage records including commands, CPU flags and dependencies, same24 inference outputs; only GPU assignment/wait order changes; no new experiment settings or DEV selection',
        supersedes='only the zero-stage waiting GPU5 dispatcher30580 of dense_stability_and_camera_v1; its GPU0 queue is unaffected',
        waits=dict(gpu0='post_cache_off_zero_v1_gpu0 controller3376',gpu4='new dense B7 tracking retrieval queue, itself after fixed-prior retrieval25648',
                   gpu5='remaining_matrix_v1_gpu5 controller12965',gpu6='native_prior_seed_pairs_v1_gpu6 controller33289'),provenance=provenance()))


if __name__=='__main__':main()

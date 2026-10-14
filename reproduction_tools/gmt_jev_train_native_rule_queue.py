"""Finite frozen native Rule validation on every registered TRAIN video."""
import collections
from gmt_jev_common import *
from gmt_jev_speed_queue import publish_step


def main():
    assert read(OUT/'opportunity_full/FOLLOWUP.json')['status'] == 'COMPLETE_FINITE_FULL_TRAIN_AUDITS'
    coverage = read(REPORTS/'FULL_TRAIN_OPPORTUNITY_AUDIT.json')
    assert coverage['status'] == 'COMPLETE_ALL_REGISTERED_TRAIN_TEMPORAL_COVERAGE'
    train = collections.Counter()
    for case in coverage['cases']:
        train.update(case['buckets']['TRAIN_BLOCKS'])
    assert train['rule_local_N01'] > train['rule_local_N10']
    queue = OUT/'train_native_rule/QUEUE.json'
    for video in [12, 13, 14]:
        name = f'P3_TRAIN{video}_rule_native_validation_GPU7'
        jobs = read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs']
        if name in jobs:
            assert jobs[name]['status'] == 'COMPLETE', 'Never duplicate a nonterminal job'
            continue
        assert not any(d['status'] == 'RUNNING' and d.get('GPU') == 7 for d in jobs.values())
        save(queue, dict(status='RUNNING_NATIVE_RULE_COMMIT_VALIDATION', PID=os.getpid(),
                         video=video, physical_GPU=7, job=name))
        subprocess.run([PYTHON, str(ROOT/'reproduction_tools/gmt_jev_job.py'),
                        '--name', name, '--gpu', '7', '--script',
                        'gmt_jev_train_native_rule_validation.py', '--', '--video', str(video)],
                       cwd=ROOT, check=True)
        publish_step(f'full TRAIN{video} frozen native Rule commit validation')
    cases = [read(REPORTS/f'FULL_TRAIN_NATIVE_RULE_VIDEO{v:02d}.json') for v in [12, 13, 14]]
    counts = collections.Counter()
    buckets = {key: collections.Counter() for key in ['TRAIN_BLOCKS', 'HELDOUT_TEMPORAL_BLOCKS']}
    for case in cases:
        assert case['status'] == 'COMPLETE_FULL_TRAIN_NATIVE_RULE_COMMIT_VALIDATION'
        counts.update(case['paired_commit_counts'])
        for key in buckets:
            buckets[key].update(case['paired_commit_buckets'][key])
    assert counts == buckets['TRAIN_BLOCKS'] + buckets['HELDOUT_TEMPORAL_BLOCKS']
    signal = buckets['TRAIN_BLOCKS']['N01'] > buckets['TRAIN_BLOCKS']['N10']
    save(REPORTS/'FULL_TRAIN_NATIVE_RULE_AUDIT.json', dict(
        status='COMPLETE_ALL_REGISTERED_TRAIN_NATIVE_RULE_VALIDATION', cases=cases,
        paired_commit_counts=dict(counts), paired_commit_buckets={k:dict(v) for k,v in buckets.items()},
        native_TRAIN_signal_positive=signal, no_gradient_updates=True,
        no_persistent_corpus=True, training_authorized_by_this_diagnostic_alone=False,
        independent_validation=False, source=binding()))
    publish_step('all registered TRAIN frozen native Rule commit validation summary')
    save(queue, dict(status='COMPLETE_FINITE_NATIVE_RULE_VALIDATION', PID=os.getpid()))


if __name__ == '__main__':
    main()

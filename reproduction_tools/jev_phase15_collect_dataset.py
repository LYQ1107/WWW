"""Freeze complete TRAIN predicted-state observations with persistent state."""
import argparse
import collections
import gzip
import time
import traceback
import torch
from jev_phase15_common import *
from jev_phase13_runtime import build_tracker, cache_inputs, run
from jev_phase14_native_risk import NativeRisk
from jev_phase14_artifacts import save_dense, load_dense
from jev_phase15_commitment_labels import certify
from audit_jev_stage2_gta_free import phase13_prefix
from gtr.modeling.jev_phase15.native_commit_adapter import attach, FrozenOriginalPolicy
from gtr.modeling.jev_native_state import fingerprint
from run_jev_phase10_closed_loop import raw_predictions


def main(video):
    protect(); storage_guard(); assert video in TRAIN
    torch.set_num_threads(1); torch.manual_seed(20261009)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    policy, checkpoint, training_result = load_old_policy('multi_question', 20261009)
    values, frames, reader = cache_inputs(video); risk = NativeRisk(video, reader)
    model = build_tracker(video, policy=FrozenOriginalPolicy(policy), react_learned=False)
    executor = attach(model)
    out = OUT/'commitment_dataset_v2'/f'video{video:02d}'; out.mkdir(parents=True, exist_ok=True)
    assert not (out/'RESULT.json').exists()
    source = binding(seed=20261009, checkpoints=[checkpoint], dataset=ref(ANNOTATIONS),
        evaluator='TRAIN-only offline certificates on actual frozen Multi mutated histories',
        scope='new persistent observations; original policy and native lifecycle unaltered')
    source['perception_input_provenance'] = perception_provenance(video)
    # Repeated annotation identities in a camera/frame are uncertifiable. This
    # masks offline labels only; downloaded GT, detections and decisions stay intact.
    duplicate_labels = 0
    for key,image in risk.labels.images.items():
        repeated=collections.Counter(a['instance_id'] for a in risk.labels.gt[image['id']])
        bad={identity for identity,n in repeated.items() if n>1}
        if bad:
            actual=risk.labels.current(*key)
            duplicate_labels+=sum(gt in bad for gt in actual if gt is not None)
            risk.labels.aligned[key]=[None if gt in bad else gt for gt in actual]
    records = []; counts = collections.Counter(); native = [None]; prefix = [None]
    expected = {}; saved_memory = [None]; start = time.monotonic(); last_progress = [0.]
    journal = gzip.open(out/'COMMITS.jsonl.gz', 'wt')

    def before_native(**d):
        native[0] = d
        if (d['frame'], d['view']) == (32, 0):
            prefix[0] = out/'NATIVE_PREFIX_32_0.pth.xz'
            save_dense(prefix[0],phase13_prefix(model,d),reserve=30*2**30)

    def before(**d):
        risk.before(**d)
        if d['task'] != 0: return
        c = d['context']; key = c['frame'], c['view']
        if not len(d['logits']): return
        labels = certify(d['batch'], d['refs'], key, risk)
        item = dict(inputs={name: value.detach().cpu().clone() for name, value in d['batch'].items()},
            key=(video, *key), refs=d['refs'].copy(), rows=list(range(len(d['logits']))), task=0,
            audit_block=(key[0]//64)%5 == 4, **labels)
        records.append(item)
        for kind in [0,1,2,3]:
            counts['commit_kind_'+str(kind)] += int((item['commit_kind'] == kind).sum())
            if item['audit_block']: counts['audit_commit_kind_'+str(kind)] += int((item['commit_kind'] == kind).sum())
        counts['WHO_multi_positive_rows'] += int((labels['positive'][:,:-1].sum(-1) > 1).sum())
        counts['trust_pure'] += int((labels['trust'] == 1).sum())
        counts['trust_polluted'] += int((labels['trust'] == 0).sum())
        if 32 <= key[0] <= 95:
            expected[key] = dict(inputs=item['inputs'], logits=d['logits'].detach().cpu().clone())

    def after(**d):
        risk.after(**d); key = d['frame'], d['view']; ids = d['instances'][-1].track_ids.cpu().tolist()
        journal.write(json.dumps(dict(key=key, ids=ids, events=d['events']))+'\n')
        if key in expected: expected[key]['ids'] = ids
        if key == (95,1): saved_memory[0] = fingerprint(executor.memory.state_dict())
        if time.monotonic()-last_progress[0] > 15:
            save(out/'PROGRESS.json', dict(status='COLLECTING_REAL_TRAIN', key=(video,*key),
                frames=frames, records=len(records), counts=dict(counts), seconds=time.monotonic()-start))
            last_progress[0] = time.monotonic()
        risk.trace.clear()

    executor.observer = before; executor.commit_observer = after
    model.jev_native_prefix_observer = before_native
    try:
        with torch.no_grad(): raw, _ = run(model, values, frames)
    finally: journal.close()
    natural = read(OUT/'train_commitment_prefixes_v1'/f'video{video:02d}'/'RESULT.json')
    assert sha(natural['raw_predictions']['path']) == natural['raw_predictions']['SHA256']
    predictions = raw_predictions(raw, risk.labels.images)
    original = read(natural['raw_predictions']['path'])
    assert predictions == original, 'instrumented state changed original committed predictions'
    assert prefix[0] is not None and saved_memory[0] is not None
    storage_guard(); artifact = out/'DATASET.pth.xz'
    save(out/'PROGRESS.json', dict(status='SERIALIZING_BEFORE_RESTORE_AUDIT', records=len(records), counts=dict(counts)))
    save_dense(artifact, dict(records=records, binding=source, native_policy='pi_multi_seed20261009_20k'), reserve=30*2**30)
    save(out/'COLLECTION.json',dict(status='COLLECTED_PENDING_NATIVE_RESTORE',binding=source,DATASET=ref(artifact),
         counts=dict(counts),raw_predictions_exact=True,duplicate_GT_offline_labels_masked=duplicate_labels))
    restored_payloads = [0]

    def verify_before(**d):
        if d['task'] != 0 or not len(d['logits']): return
        key = d['context']['frame'], d['context']['view']; old = expected[key]
        differences={name:float((value.cpu().float()-old['inputs'][name].float()).abs().max())
            for name,value in d['batch'].items() if not torch.equal(value.cpu(),old['inputs'][name])}
        assert not differences, (key,differences)
        assert torch.equal(d['logits'].cpu(), old['logits'])

    def verify_after(**d):
        key = d['frame'], d['view']
        if key in expected: assert d['instances'][-1].track_ids.cpu().tolist() == expected[key]['ids']
        restored_payloads[0] += 1

    executor.observer = verify_before; executor.commit_observer = verify_after
    model.jev_native_prefix_observer = None
    restored_prefix=load_dense(prefix[0],map_location='cuda:0')
    with torch.no_grad(): run(model, values, frames, stop=95, prefix=restored_prefix)
    assert fingerprint(executor.memory.state_dict()) == saved_memory[0]
    assert restored_payloads[0] == 128
    result = dict(status='PASS', binding=source, video=video, frames=frames, records=len(records),
        counts=dict(counts), DATASET=ref(artifact), original_native_source=ref(OUT/'train_commitment_prefixes_v1'/f'video{video:02d}'/'RESULT.json'),
        exact_original_native_predictions=True, persistent_restore_payloads_exact=restored_payloads[0],
        persistent_restore_inputs_logits_IDS_and_final_memory_exact=True,
        GTA_free_throw='PASS_ACTUAL_NATIVE_EXECUTION', GT_actor_inputs=False, teacher_forced_IDs=False,
        reserved_TRAIN_blocks_never_gradients=True, seconds=time.monotonic()-start,
        duplicate_GT_offline_labels_masked=duplicate_labels,raw_GT_annotations_modified=False,
        correction_semantics='strong causal predecessor certified pure wrong with clean compatible alternatives; mixed predecessor alone remains UNKNOWN')
    save(out/'RESULT.json', result); save(out/'PROGRESS.json', dict(status='PASS', records=len(records), counts=dict(counts)))
    print('PHASE15_COMMITMENT_DATASET_PASS', video, len(records), dict(counts), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--video',type=int,required=True); a=parser.parse_args()
    try: main(a.video)
    except Exception: failure('commitment_dataset',traceback.format_exc()); raise

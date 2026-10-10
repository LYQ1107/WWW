"""One bounded TRAIN collection under the frozen second pilot's own native policy."""
import argparse
import collections
import gzip
import lzma
import time
import torch
from jev_phase15_common import *
from jev_phase15_evaluate import load_policy
from jev_phase15_commitment_labels import certify
from jev_phase15_recent_labels import patch_record
from jev_phase13_runtime import build_tracker,cache_inputs,run
from jev_phase14_native_risk import NativeRisk
from jev_phase14_artifacts import load_dense
from audit_jev_stage2_gta_free import phase13_prefix
from gtr.modeling.jev_phase15.native_commit_adapter import attach
from gtr.modeling.jev_native_state import fingerprint
from run_jev_phase10_closed_loop import raw_predictions


def fast_dense(path,value):
    path=Path(path);assert not path.exists();temporary=path.with_name(path.name+'.tmp')
    with lzma.open(temporary,'wb',preset=1) as stream:torch.save(value,stream)
    storage_guard();temporary.replace(path)


def main(video):
    protect();storage_guard();assert video in TRAIN
    protocol=read(REPORTS/'RESEARCH_VERSION_3_PROTOCOL.json');assert protocol['status']=='FROZEN_BEFORE_TRAINING'
    torch.set_num_threads(1);torch.manual_seed(20261009);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    policy,checkpoint,trained=load_policy('F_full',20261009,2,'pilot')
    assert checkpoint==protocol['collector_checkpoint']
    values,frames,reader=cache_inputs(video);model=build_tracker(video,policy,react_learned=False);executor=attach(model)
    risk=NativeRisk(video,reader)
    duplicate_labels=0
    for key,image in risk.labels.images.items():
        repeated=collections.Counter(a['instance_id'] for a in risk.labels.gt[image['id']]);bad={gt for gt,n in repeated.items() if n>1}
        if bad:
            targets=risk.labels.current(*key);duplicate_labels+=sum(gt in bad for gt in targets if gt is not None)
            risk.labels.aligned[key]=[None if gt in bad else gt for gt in targets]
    out=OUT/'onpolicy_pilot_dataset_v3_r2'/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True);assert not (out/'RESULT.json').exists()
    source=binding(seed=20261009,checkpoints=[checkpoint],dataset=ref(ANNOTATIONS),evaluator='past-certified labels on frozen v2 actual native mutated histories',
        scope='one TRAIN-only diagnostic pilot corpus collection; no gradient, no ideal/GT native state')
    source['perception_input_provenance']=perception_provenance(video)
    records=[];counts=collections.Counter();recent=collections.defaultdict(lambda:collections.deque(maxlen=3));expected={};memory=[None];prefix=[None];last=[0.];start=time.monotonic()
    journal=gzip.open(out/'COMMITS.jsonl.gz','wt')
    def before_native(**d):
        if (d['frame'],d['view'])==(32,0):
            prefix[0]=out/'NATIVE_PREFIX_32_0.pth.xz';fast_dense(prefix[0],phase13_prefix(model,d))
        if (d['frame'],d['view'])==(96,0):
            # The previous observer ran before the adapter's posterior write.
            # This boundary sees the complete actual post-commit state of95/1.
            memory[0]=fingerprint(executor.memory.state_dict())
    def before(**d):
        risk.before(**d)
        if d['task']!=0 or not len(d['logits']):return
        c=d['context'];key=c['frame'],c['view'];labels=certify(d['batch'],d['refs'],key,risk)
        r=dict(inputs={k:v.detach().cpu().clone() for k,v in d['batch'].items()},key=(video,*key),refs=d['refs'].copy(),
            rows=list(range(len(d['logits']))),task=0,audit_block=(key[0]//64)%5==4,corpus_actor='F_pilot_v2',**labels)
        patch,statistics=patch_record(r,recent);r.update(patch);records.append(r)
        split='audit' if r['audit_block'] else 'train'
        for kind in [0,1,2,3]:counts[f'{split}_commit_kind_{kind}']+=int((r['commit_kind']==kind).sum())
        counts['recent_certificate_added']+=statistics['additional_necessary_correction']
        f=r['inputs']['commitment_features'];counts['nonconstant_posterior_payloads']+=int(bool((f[...,16]!=.5).any() or (f[...,17]!=0).any()))
        if 32<=key[0]<=95:expected[key]=dict(inputs=fingerprint(d['batch']),logits=fingerprint(d['logits']))
    def after(**d):
        risk.after(**d);key=d['frame'],d['view'];ids=d['instances'][-1].track_ids.cpu().tolist()
        journal.write(json.dumps(dict(key=key,ids=ids,events=d['events']))+'\n')
        targets=risk.labels.current(*key)
        assert len(ids)==len(targets)
        for identity,gt in zip(ids,targets):recent[identity,key[1]].append((key[0],gt))
        if key in expected:expected[key]['ids']=ids
        if time.monotonic()-last[0]>=15:
            save(out/'PROGRESS.json',dict(status='COLLECTING_ACTUAL_V2_NATIVE',key=(video,*key),frames=frames,records=len(records),counts=dict(counts),seconds=time.monotonic()-start));journal.flush();last[0]=time.monotonic()
        risk.trace.clear()
    executor.observer=before;executor.commit_observer=after;model.jev_native_prefix_observer=before_native
    try:
        with torch.no_grad():raw,_=run(model,values,frames)
    finally:journal.close()
    save(out/'RAW_PREDICTIONS.json',raw_predictions(raw,risk.labels.images));assert memory[0] is not None
    artifact=out/'DATASET.pth.xz';save(out/'PROGRESS.json',dict(status='LOSSLESS_SERIALIZATION_PRESET1',records=len(records),counts=dict(counts)))
    fast_dense(artifact,dict(records=records,binding=source,native_policy='actual_F_pilot_v2'))
    restored=[0]
    def verify_before(**d):
        if d['task']==0 and len(d['logits']):
            old=expected[d['context']['frame'],d['context']['view']]
            assert fingerprint(d['batch'])==old['inputs'] and fingerprint(d['logits'])==old['logits']
    def verify_after(**d):
        key=d['frame'],d['view']
        if key in expected:assert d['instances'][-1].track_ids.cpu().tolist()==expected[key]['ids']
        restored[0]+=1
    executor.observer=verify_before;executor.commit_observer=verify_after;model.jev_native_prefix_observer=None
    with torch.no_grad():run(model,values,frames,stop=95,prefix=load_dense(prefix[0],map_location='cuda:0'))
    assert restored[0]==128 and fingerprint(executor.memory.state_dict())==memory[0]
    result=dict(status='PASS',binding=source,video=video,frames=frames,records=len(records),counts=dict(counts),DATASET=ref(artifact),
        actual_mutated_state=True,own_frozen_v2_policy=True,GT_actor=False,no_optimizer_updates=True,
        persistent_restore_payloads_exact=restored[0],full_inputs_logits_IDs_and_commitment_memory_exact=True,
        raw_predictions=ref(out/'RAW_PREDICTIONS.json'),commits=ref(out/'COMMITS.jsonl.gz'),
        duplicate_GT_offline_labels_masked=duplicate_labels,raw_GT_unchanged=True,lossless_xz_preset=1,
        reserved_audit_split_unchanged=True,seconds=time.monotonic()-start)
    save(out/'RESULT.json',result);save(out/'PROGRESS.json',dict(status='PASS',records=len(records),counts=dict(counts)))
    print('PHASE15_PILOT_ONPOLICY_COLLECTION',video,dict(counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);a=p.parse_args();main(a.video)

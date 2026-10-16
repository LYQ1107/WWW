"""FIT12 precommit opportunity audit on real full native matrices, CPU only.

This is original-state policy replay, not online HOTA or veto utility. Complete
GT-free policy events are sealed before offline supervision is interpreted.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time
import torch
from scipy.optimize import linear_sum_assignment
from common import ROOT,sha256,save_json
from cir_native_fit_inputs import FIT,all_calls
from gtr.modeling.identity_revision.features import c1_scores
from gtr.modeling.identity_revision.native_evidence import native_conflicts
from gtr.modeling.identity_revision.conflict_detector import FixedConflictDetector


def solve(scores,ids,thresholds):
    rows,columns=linear_sum_assignment(-scores.numpy());out=torch.full((len(scores),),-1,dtype=torch.long)
    for row,col in zip(rows,columns):
        if scores[row,col]>thresholds[col]:out[row]=ids[col]
    return out


def read_rows(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:yield json.loads(line)


def offline_labels():
    path=Path('/data1/liuyeqiang/WWW_pirga_runtime_20261016/fulltrain_admission_v1/offline_supervision.jsonl.gz')
    if sha256(path)!='4cefb1f4af6e594b03ac57d8e47596778209eaed99f414f84c6d0486ce729014':raise ValueError('Separate label source changed')
    labels={tuple(row['event_key']):row for row in read_rows(path)}
    root=Path('/data1/liuyeqiang/WWW_trackvggt_runtime/cv_pil_dev12_native_release_v1')
    seal=json.loads((root/'artifact_sha256.json').read_text())
    path=root/'offline_labels/queries.jsonl.gz'
    if sha256(path)!=seal['artifacts']['offline_labels/queries.jsonl.gz']:raise ValueError('TRAIN12 label source changed')
    for row in read_rows(path):
        ids=sorted(map(int,row['pure_track_gt']))
        labels[tuple(row['key'])]=dict(GT_person=row['gt_identity'],candidate_persons=[row['pure_track_gt'][str(i)] for i in ids],candidate_ids=ids)
    return labels


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--protocol-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if sha256(a.protocol)!=a.protocol_sha:raise ValueError('Unpinned bounded FIT policy audit')
    protocol=json.loads(a.protocol.read_text())
    if protocol['fit_video_ids']!=list(FIT) or protocol['GT_may_be_read_before_policy_seal']:raise ValueError('Scope/GT separation changed')
    detector=FixedConflictDetector()
    if protocol['conflict_rule']!=detector.protocol():raise ValueError('Unregistered rule change')
    for name,digest in protocol['source_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Audit source drift '+name)
    coverage=json.loads((ROOT/'results/CIR_GMT/TRAIN24_COVERAGE.json').read_text())
    a.out.mkdir(parents=True);torch.set_num_threads(1);start=time.monotonic();counts=Counter();scene_counts=defaultdict(Counter)
    events=a.out/'FIT_POLICY_EVENTS.jsonl.gz'
    with gzip.open(events,'xt') as stream:
        for call in all_calls(coverage):
            if time.monotonic()-start>protocol['maximum_CPU_wall_seconds']:raise RuntimeError('Registered CPU wall limit')
            native=solve(call['native_scores'],call['candidates'],call['thresholds'])
            if native.tolist()!=call['original_ids']:raise ValueError('Full original joint native replay mismatch')
            scores=c1_scores(call['native_scores'],call['q'],call['h'],call['history_ids'],call['candidates'],call['history_frames'],call['frame'])
            accepted=solve(scores,call['candidates'],call['thresholds'])
            evidence,features=native_conflicts(call['q'],call['qboxes'],call['frame'],call['camera'],call['h'],call['hboxes'],call['history_ids'],
                call['history_frames'],call['history_cameras'],call['candidates'],scores,accepted,call['physical'])
            for row,key in enumerate(call['keys']):
                state,reasons=detector.classify(evidence[row]);counts[state.value]+=1;scene_counts[str(call['scene'])][state.value]+=1
                ev=asdict(evidence[row]);ev={k:(v if not isinstance(v,float) or __import__('math').isfinite(v) else None) for k,v in ev.items()}
                event=dict(key=key,frame=call['frame'],camera=call['camera'],candidate_ids=call['candidates'].tolist(),
                    original_primary_id=int(native[row]),C1_primary_id=int(accepted[row]),risk_features=features[row].tolist(),
                    evidence_state=state.value,evidence=ev,reasons=reasons,precision=call['precision'])
                stream.write(json.dumps(event,separators=(',',':'),allow_nan=False)+'\n')
            counts['queries']+=len(call['keys']);counts['calls']+=1
            if counts['calls']%250==0:print(json.dumps(dict(calls=counts['calls'],queries=counts['queries'],scene=call['scene'],GT_semantics_loaded=False,seconds=time.monotonic()-start)),flush=True)
    seal=sha256(events)
    save_json(a.out/'POLICY_SEAL.json',dict(status='SEALED_GT_FREE_POLICY_OUTPUTS',events_sha256=seal,counts=dict(counts),protocol_sha256=a.protocol_sha))
    # No label identity has been parsed above this boundary.
    labels=offline_labels();training=a.out/'OFFLINE_RISK_TRAINING.jsonl.gz';outcomes=Counter();groups=defaultdict(set);per_scene=defaultdict(Counter)
    with gzip.open(training,'xt') as stream:
        for event in read_rows(events):
            key=tuple(event['key']);label=labels[key];person=label['GT_person'];ids=event['candidate_ids']
            if 'candidate_ids' in label and label['candidate_ids']!=ids:raise ValueError('GT tried to change candidate set')
            if len(label['candidate_persons'])!=len(ids):raise ValueError('Offline label candidate order mismatch')
            selected=event['C1_primary_id'];owner=label['candidate_persons'][ids.index(selected)] if selected in ids else None
            risk=None if person is None or owner is None else int(owner!=person)
            tag='UNKNOWN' if risk is None else ('WRONG_ACCEPT' if risk else 'CORRECT_ACCEPT')
            outcomes[tag]+=1;per_scene[str(key[0])][tag]+=1
            if risk is not None:groups[tag].add((key[0],person))
            if event['evidence_state']=='SUSPECT':
                outcomes['SUSPECT_'+tag]+=1
                if risk is not None:groups['SUSPECT_'+tag].add((key[0],person))
            stream.write(json.dumps(dict(key=list(key),features=event['risk_features'],risk_label=risk,person=person,
                source_state='Original GMT history + same-context C1 primary joint proposal, no mutated future/veto utility yet'),separators=(',',':'),allow_nan=False)+'\n')
    enough=outcomes['WRONG_ACCEPT']>=protocol['minimum_wrong_queries_for_learning'] and len(groups['WRONG_ACCEPT'])>=protocol['minimum_wrong_person_groups_for_learning'] and len({s for s,p in groups['WRONG_ACCEPT']})>=protocol['minimum_wrong_scenes_for_learning']
    result=dict(status='COMPLETE_FIT12_ORIGINAL_STATE_GUARD_OPPORTUNITY_AUDIT',fit_video_ids=list(FIT),counts=dict(counts),by_scene_policy={s:dict(c) for s,c in scene_counts.items()},
        offline_outcomes=dict(outcomes),known_person_dependence_groups={tag:len(g) for tag,g in groups.items()},
        by_scene_offline={s:dict(c) for s,c in per_scene.items()},potential_learning_supervision='ENOUGH_FOR_BOUNDED_BINARY_RISK_CLASSIFICATION' if enough else 'INSUFFICIENT_CERTIFIED_RISK_SUPERVISION',
        veto_causal_utility_qualified=False,actual_Guard_online_gain=None,
        interpretation='GT-free rule outputs precede labels. A wrong precommit edge is not automatically a beneficial veto; true paired mutated futures and complete video remain required.',
        features_file=dict(path=str(training),sha256=sha256(training)),policy_events=dict(path=str(events),sha256=seal),
        protocol_sha256=a.protocol_sha,CPU_wall_seconds=time.monotonic()-start,GPU_wall_hours=0)
    save_json(a.out/'RESULT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()

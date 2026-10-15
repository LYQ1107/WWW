"""P1 read-only source admission and coverage, separate from model evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from cv_pil_dev12_assets import require
from cv_pil_dev12_shards import NativeShardAssets, rows
from cv_pil_native_handoff import sha256


def execute(pins, out):
    require(not out.exists(), 'Existing admission cannot be overwritten');started=time.monotonic()
    release=pins['release'];a=NativeShardAssets(**release)
    require(not a.engineering_only, 'Full native release required')
    labels=a.supervision();n=len(labels);errors=[];reachable=[];head_reachable=[];people=set();camera_groups=set()
    tag_counts=Counter();unknown=covered=0
    for k,s in labels.items():
        raw=s['source_label'];person=s['GT_person'];cid,row=a.query_index[k];c=a.calls[cid]
        qo=a.observations[k[1:]]
        unknown+=person is None;covered+=bool(s['certified_target_indices']);tag_counts.update(raw['tags'])
        if person is not None:people.add((12,person));camera_groups.add((12,person,qo['camera']))
        if raw['known_original_wrong']:
            errors.append(k)
            if s['certified_target_indices']:
                reachable.append(k);positive={c['track_ids'][i] for i in s['certified_target_indices']}
                if any(r['track'] in positive and a.observations[tuple(r['source'])]['frame']<qo['frame'] for r in c['references']):head_reachable.append(k)
    original=Path(pins['original_audit_manifest']);require(sha256(original)==pins['original_audit_manifest_sha256'], 'Original audit manifest differs')
    source=json.loads(original.read_text());root=a.root
    require(sha256(root/pins['prediction_role'])==source['predictions_sha256'], 'Full original prediction bytes differ')
    native=list(rows(root/pins['decision_role']));baseline=list(rows(original.parent/'actual_decisions.jsonl.gz'))
    require(native==baseline, 'Original full primary/Bank/final-ID trace differs')
    del native,baseline
    result=dict(status='PASS_FULL_NATIVE_DEV12_INPUTS',coverage=dict(total_queries=n,full1152_query_features=n,
        complete_native_scores=n,complete_legal_histories=n,UNKNOWN_queries=unknown,
        positive_covered_queries=covered,certified_original_primary_errors=len(errors),
        legal_positive_reachable_errors=len(reachable),strict_past_head_input_eligible_errors=len(head_reachable),
        current_interface_certified_correctable_errors=None,unique_video_person_groups=len(people),
        unique_video_person_camera_groups=len(camera_groups),label_tags=dict(tag_counts)),
        source_primary_error_keys=[list(k) for k in errors],
        scope_sha256=release['scope_sha'],manifest_sha256=release['manifest_sha'],seal_sha256=release['seal_sha'],
        original_full_prediction_bytes_exact=True,original_full_decision_trace_exact=True,
        every_full_primary_and_Bank_Hungarian_replayed=True,every_physical_final_commit_linked=True,
        native_descriptor_versions=len(a.variants),cpu_aggregation_max_abs_error=a.max_aggregation_error,
        full_CV_PIL_controller_OFF_parity='NOT_RUN_NO_ONLINE_MODULE_INTEGRATED',
        original_exporter_real40_OFF_ON_state_parity='SEPARATELY_VERIFIED',
        exposed_development_not_blind=True,training=False,model_inference=False,GT_in_actor=False,
        true_final_counterfactual_identity_fixes=None,wall_seconds=time.monotonic()-started,gpu_wall_hours=0,
        scientific_success=False)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['coverage']),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--pins',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);execute(json.loads(a.pins.read_text()),a.out)


if __name__=='__main__':main()

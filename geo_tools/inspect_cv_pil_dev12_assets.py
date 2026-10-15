"""Admit an externally pinned development release, without fitting any model."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from cv_pil_dev12_assets import Dev12Assets
from cv_pil_native_handoff import sha256


def coverage(assets):
    labels = assets.supervision()
    errors = reachable = actor_reachable = unknown = covered = 0
    people = set(); person_camera = set(); episodes = set(); states = Counter()
    same_frame = 0
    for k, s in labels.items():
        actor = assets.actor(k)
        person = s['GT_person']; positives = s['certified_target_indices']
        states[s['primary_state']] += 1
        unknown += person is None; covered += bool(positives)
        if person is not None:
            people.add((12, person)); person_camera.add((12, person, actor['camera']))
            episodes.add((12, person, actor['camera'], (actor['frame'] - 1)//50))
        is_error = s['primary_state'] in ('wrong_link', 'missed_known_positive')
        errors += is_error; reachable += is_error and bool(positives)
        actor_reachable += is_error and any(actor['past_histories'][i] for i in positives)
        c = assets.calls[assets.query_index[k][0]]
        same_frame += sum(assets.obs[tuple(r)]['frame'] == actor['frame'] for r in c['reference_keys'])
    n = len(labels)
    return dict(total_queries=n, full_1152_query_features=n, complete_native_scores=n,
                complete_legal_histories=n, query_unknown=unknown, positive_covered=covered,
                certified_primary_errors=errors, legal_positive_reachable_errors=reachable,
                strict_past_head_input_eligible_errors=actor_reachable,
                current_interface_certified_correctable_errors=None,
                correctability_note='Input reachability is necessary, not a full native repair proof.',
                unique_video_person_groups=len(people), unique_video_person_camera_groups=len(person_camera),
                fixed_50_frame_person_camera_time_groups=len(episodes), primary_states=dict(states),
                same_frame_reference_query_pairs_excluded_only_from_old_heads=same_frame,
                actual_counterfactual_committed_fixes=None, actual_counterfactual_committed_harms=None,
                exposed_development_not_independent=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--pins', type=Path, required=True,
                   help='Independently sealed provenance and scope; not copied from an unverified release')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    if a.out.exists(): raise FileExistsError(a.out)
    pins = json.loads(a.pins.read_text())
    assets = Dev12Assets(a.manifest, a.manifest_sha256, pins['provenance'], a.root, pins['scope'])
    report = dict(status='INPUTS_VALIDATED_OLD_CHECKPOINT_EVALUATION_NEXT',
                  manifest_sha256=a.manifest_sha256, independent_pins_sha256=sha256(a.pins),
                  coverage=coverage(assets), models_trained=False, gpu_wall_hours=0,
                  full_native_OFF_parity='NOT_RUN', scientific_success=False)
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(a.out)


if __name__ == '__main__': main()

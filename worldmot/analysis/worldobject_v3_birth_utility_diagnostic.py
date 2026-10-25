"""Synthetic actual-decision probes of frozen R2; not a replacement solver."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from worldobject_synthetic import core, obs, gid
from test_worldobject_memory import evidence


def one_case(name, *, delta=1e-6, birth_logit=-40., history=False,
             geometry=True, conflict=False, competing=False):
    tracker = core()
    if history:
        tracker.step([obs(1)], 0)
    timestamp = 1 if history else 0
    observations = [obs(1, timestamp=timestamp, geometry=geometry),
                    obs(2, timestamp=timestamp, geometry=geometry,
                        center=(20, 0, 3) if conflict else (0, 0, 3), feature=(0., 1., 0.))]
    labels = ["PERSON_A", "PERSON_B"]
    if competing:
        observations = [obs(1, 0), obs(1, 1), obs(2, 0), obs(2, 1)]
        labels = ["PERSON_A", "PERSON_B", "PERSON_C", "PERSON_D"]
    n = len(observations)
    learned = evidence(1 if history else 0, n)
    learned["birth_logits"][:] = birth_logit
    threshold_logit = np.log(tracker.config.birth_pair_threshold / (1-tracker.config.birth_pair_threshold))
    learned["birth_pair_logits"] = np.full((n, n), -30.)
    for left in range(n):
        for right in range(left + 1, n):
            if observations[left].camera_id != observations[right].camera_id:
                learned["birth_pair_logits"][left, right] = threshold_logit + delta
                learned["birth_pair_logits"][right, left] = threshold_logit + delta
    if history:
        learned["assignment_logits"][:] = -8.
        learned["assignment_logits"][0, 0] = 8.
        learned["null_logits"][:] = -2.
    result = tracker.step(observations, timestamp, evidence=learned)
    ids = [gid(result, item.camera_id, item.detection_id) for item in observations]
    decisions = {(x["camera_id"], x["detection_id"]): x["decision"] for x in result["decisions"]}
    cross = [(a,b) for a in range(n) for b in range(a+1,n)
             if observations[a].camera_id != observations[b].camera_id]
    selected = [(a,b) for a,b in cross if ids[a] == ids[b]]
    return {"name": name, "pair_logit_minus_threshold": delta,
        "singleton_birth_logit": birth_logit, "history_competitor": history,
        "geometry_reliable": geometry, "geometry_conflict": conflict,
        "raw_above_threshold_wrong_pairs": int(sum(learned["birth_pair_logits"][a,b] > threshold_logit for a,b in cross)),
        "solver_birth_proposals": len(tracker.bank.assigner.last_birth_candidate_edges),
        "selected_cross_view_pairs": selected,
        "actual_wrong_global_id_merges": sum(labels[a] != labels[b] for a,b in selected),
        "delivered_global_ids": ids, "birth_objects": result["bank"]["births"],
        "actual_decisions": [decisions[item.key] for item in observations],
        "bank_constraints_satisfied": result["bank"]["constraints_satisfied"]}


def run_scenarios():
    return [one_case("JUST_BELOW_THRESHOLD", delta=-1e-6),
            one_case("EXACT_THRESHOLD_ZERO_BENEFIT", delta=0.),
            one_case("TINY_POSITIVE_PAIR_NO_HISTORY", delta=1e-6),
            one_case("LOW_SINGLETON_PRIOR_NO_HISTORY", delta=1e-6, birth_logit=-60.),
            one_case("CORRECT_HISTORY_COMPETES", history=True),
            one_case("MULTIPLE_WRONG_PAIRS_COMPETE", competing=True),
            one_case("RELIABLE_GEOMETRY_VETO", conflict=True),
            one_case("SAME_PAIR_WEAK_GEOMETRY", geometry=False, conflict=True)]


def spatial_recovery_probe(method):
    tracker=core(method)
    def learned(m,n):
        result=evidence(m,n);result['assignment_logits'][:]=8.;result['null_logits'][:]=-2.
        result['birth_logits'][:]=-10.;result['birth_pair_logits']=np.full((n,n),10.)
        return result
    tracker.step([obs(1),obs(2)],0,evidence=learned(0,2))
    tracker.step([],1,evidence=learned(1,0))
    recovered=tracker.step([obs(1,center=(20,0,3),timestamp=2),obs(2,center=(20,0,3),timestamp=2)],
                           2,evidence=learned(1,2))
    previous=tracker.bank.objects[1]
    retained_center=previous.center.tolist() if previous.geometry_valid else None
    after_missed=previous.missed
    following=tracker.step([obs(1,center=(20,0,3),timestamp=3),obs(2,center=(20,0,3),timestamp=3)],
                           3,evidence=learned(1,2))
    return {'method':method,'recovered_global_ids':[gid(recovered,1),gid(recovered,2)],
        'next_frame_global_ids':[gid(following,1),gid(following,2)],
        'center_retained_after_recovery':retained_center,'missed_after_recovery':after_missed,
        'recovery_decisions':[x['decision'] for x in recovered['decisions']],
        'next_frame_decisions':[x['decision'] for x in following['decisions']],
        'both_views_always_same_clean_person':True,'appearance_exactly_identical':True,
        'current_position_unchanged_between_recovery_and_next_frame':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True,type=Path);args=parser.parse_args()
    result={"status":"FROZEN_R2_SYNTHETIC_ACTUAL_DECISION_DIAGNOSIS",
        "diagnostic_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases":run_scenarios(),"spatial_recovery_probes":[spatial_recovery_probe(method) for method in
            ('FULL_WORLD_OBJECT_V3','HEAD_ONLY_NO_GEOMETRY_V3','FULL_NO_GEOMETRY_V3')],
        "TRAIN_or_TEST_data_used":False,"R2_source_modified":False,
        "new_training_started":False,"R3_implemented_or_started":False,
        "conclusion":"A geometry-admissible wrong pair with any strictly positive utility can beat two zero-utility singleton births when no historical option competes. A correct historical competitor or reliable spatial conflict can prevent that merge. This proves a structural synthetic possibility, not its frequency or net impact on real TRAIN VAL."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({"cases":len(result["cases"]),"output":str(args.output)}))


if __name__=="__main__":main()

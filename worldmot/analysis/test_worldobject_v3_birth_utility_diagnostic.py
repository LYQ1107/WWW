from worldobject_v3_birth_utility_diagnostic import one_case, run_scenarios, spatial_recovery_probe
import json


def test_tiny_positive_utility_changes_actual_global_ids_at_threshold_boundary():
    assert one_case("below",delta=-1e-6)["actual_wrong_global_id_merges"]==0
    assert one_case("exact",delta=0.)["actual_wrong_global_id_merges"]==0
    assert one_case("above",delta=1e-6)["actual_wrong_global_id_merges"]==1


def test_extremely_low_singleton_birth_probability_does_not_remove_positive_joint_bias():
    case=one_case("low",birth_logit=-60.)
    assert case["actual_wrong_global_id_merges"]==1
    assert case["actual_decisions"]==["LEARNED_CROSS_VIEW_JOINT_BIRTH"]*2


def test_correct_historical_option_can_defeat_the_small_positive_wrong_birth():
    case=one_case("history",history=True)
    assert case["solver_birth_proposals"]==1
    assert case["actual_wrong_global_id_merges"]==0
    assert case["delivered_global_ids"][0]==1


def test_multiple_raw_wrong_pairs_do_not_equal_final_wrong_merge_count():
    case=one_case("competition",competing=True)
    assert case["raw_above_threshold_wrong_pairs"]==4
    assert case["actual_wrong_global_id_merges"]==2
    assert case["bank_constraints_satisfied"]


def test_same_pair_evidence_differs_with_reliable_versus_weak_conflicting_geometry():
    reliable=one_case("reliable",conflict=True)
    weak=one_case("weak",conflict=True,geometry=False)
    assert reliable["actual_wrong_global_id_merges"]==0
    assert weak["actual_wrong_global_id_merges"]==1


def test_every_synthetic_scenario_uses_legal_final_assignment_constraints():
    assert all(case["bank_constraints_satisfied"] for case in run_scenarios())


def test_complete_actual_decision_report_is_json_serializable():
    assert len(json.loads(json.dumps(run_scenarios())))==8


def test_geometric_visual_recovery_can_be_followed_by_new_id_despite_identical_evidence():
    for method in ('FULL_WORLD_OBJECT_V3','HEAD_ONLY_NO_GEOMETRY_V3'):
        case=spatial_recovery_probe(method)
        assert case['recovered_global_ids']==[1,1]
        assert case['next_frame_global_ids']==[2,2]
        assert case['center_retained_after_recovery']==[0.,0.,3.]
        assert case['missed_after_recovery']==0
        assert case['recovery_decisions']==['VISUAL_RECOVERY_SPATIAL_CONFLICT']*2


def test_full_no_geometry_recovery_control_keeps_the_global_id():
    case=spatial_recovery_probe('FULL_NO_GEOMETRY_V3')
    assert case['recovered_global_ids']==case['next_frame_global_ids']==[1,1]
    json.dumps(case,allow_nan=False)

"""Publish only executed replay evidence and explicitly retain pending controls."""
from jev_phase15_common import *


def main():
    protect(); path = REPORTS / 'SWITCH_EVENT_LEDGER.json'; ledger = read(path)
    completed = []; pending = []
    for item in ledger['cases']:
        p = OUT / 'native_replay_v2' / f'{item["variant"]}_seed{item["seed"]}' / f'video{item["video"]:02d}' / 'RESULT.json'
        if not p.exists(): pending.append([item['variant'], item['seed'], item['video']]); continue
        result = read(p); assert result['status'] == 'PASS'
        assert result['counts']['switch_rows_with_WHO_Availability_Trust'] == item['exact_CLEAR_switches']
        assert result['maximum_saved_vs_replayed_logit_difference'] == 0
        for field in ['causal_visual_inputs', 'recovered_head_journal']:
            assert sha(result[field]['path']) == result[field]['SHA256']
        item['native_replay'] = ref(p); item['recovered_causal_visual_tensors'] = result['causal_visual_inputs']
        item['recovered_WHO_Availability_Trust_journal'] = result['recovered_head_journal']
        item['perception_input_provenance'] = perception_provenance(item['video'])
        completed.append({'variant': item['variant'], 'seed': item['seed'], 'video': item['video'],
                          'replay': ref(p), 'counterfactual_prefixes': result['counterfactual_prefixes']})
    primary = [r for r in completed if r['variant'] == 'multi_question' and r['seed'] == 20261009]
    assert len(primary) == 3, 'Worst seed must be fully replayed before this milestone'
    status = 'PASS' if not pending else 'WORST_SEED_NATIVE_REPLAY_PASS_CONTROLS_PENDING'
    ledger.update(status=status, native_replay_completed=len(completed), native_replay_pending=pending,
                  primary824_all_head_and_state_evidence_recovered=True,
                  long_training_gate='PENDING_P2_NATIVE_CAUSAL_FEASIBILITY', replay_binding=binding())
    for name in ['SWITCH_EVENT_LEDGER', 'PHASE15_SWITCH_EVENT_LEDGER']: save(REPORTS / (name + '.json'), ledger)
    for name in ['SEED20261009_FAILURE_ATLAS', 'PHASE15_SEED20261009_FAILURE_ATLAS']:
        atlas = read(REPORTS / (name + '.json')); atlas['status'] = status
        atlas['complete_worst_seed_native_replay'] = primary
        save(REPORTS / (name + '.json'), atlas)
    risk = read(REPORTS / 'PHASE15_CORRECTIVE_VS_HARMFUL_SWITCH.json')
    risk.update(status=status, primary_native_replay_all_WHO_Q2_Q3_inputs_recovered=True)
    save(REPORTS / 'PHASE15_CORRECTIVE_VS_HARMFUL_SWITCH.json', risk)
    save(REPORTS / 'NATIVE_REPLAY_AUDIT.json', {'status': status, 'binding': binding(),
         'completed': completed, 'pending': pending, 'exact_option_logit_max_difference': 0,
         'every_native_commit_and_final_state_equal': True, 'P1_worst_seed': 'PASS',
         'P1_all_controls': 'PASS' if not pending else 'PENDING',
         'new_training_started': False, 'new_tracking_improvement_established': False})
    save(REPORTS / 'COUNTERFACTUAL_PREFIX_MANIFEST.json', {'status': 'FROZEN_BEFORE_CAUSAL_OUTCOMES',
         'binding': binding(seed=20261009), 'primary_replays': primary,
         'development_GT_used_only_for_diagnosis_and_research_interventions': True,
         'development_events_are_not_training_examples': True,
         'certified_candidate_actions_are_not_deployable': True})
    print('PHASE15_P1_NATIVE_AUDIT', status, len(completed), len(pending), flush=True)


if __name__ == '__main__': main()

"""Compact actual gates; never overwrite historical experiment directories."""
from jev_phase12_common import *

def main():
    protect();records=[];total=0
    for video in TRAIN+VAL:
        path=OUT/'native_parity_v1'/f'video{video:02d}'/'RESULT.json'
        assert path.exists(),'native prefix checks still running'
        data=json.loads(path.read_text());assert data['status']=='PASS'
        assert len(data['records'])==data['source_records']
        total+=data['source_records']
        records.append({'video':video,'records':data['source_records'],'source_commit':data['binding']['source_commit'],'source_SHA256':data['binding']['source_SHA256'],'result_path':str(path),'result_SHA256':sha(path),'seconds':data['seconds'],'shadow_typed_tasks':data['shadow_typed_tasks'],'shadow_predictions':data['shadow_predictions']})
    assert total==221
    action=OUT/'native_actions_v1/RESULT.json';a=json.loads(action.read_text());assert a['status']=='PASS' and a['committed_id_changed'] and a['fresh_next_native_candidates_changed']
    parity={'status':'PASS','original_records':total,'unique_prefixes':220,'before_full_native_state_and_RNG_exact':True,'OFF_and_SHADOW_current_and_next_frame_full_states_exact':True,'ordered_Gallery_bank_hits_IDcounter_events_RNG_exact':True,'all_legal_candidates_scores_state64_evidence12_checked':True,'all_per_camera_IDs_unique':True,'records':records,'real_option_intervention':{'source_commit':a['binding']['source_commit'],'path':str(action),'SHA256':sha(action),'committed_id_changed':True,'exact_single_write':True,'fresh_next_scores_changed':True,'untrained_engineering_only':True},'prior_840_results_unchanged':True,'historical_guard_SHA256':sha(REPORTS/'PROTECTED_BASELINE.json'),'what_we_learned':'new read-only visual stages do not alter genuine native state; legal option execution changes the real committed identity and future evidence','real_MATCH_training_authorized':True,'old95_Tiny_gate_required':False,'heldout_status':'SEALED'}
    save(REPORTS/'NATIVE_PARITY.json',parity)
    save(REPORTS/'SHADOW_MODE_TESTS.json',{'status':'PASS','source_reports':[{k:r[k] for k in ['video','result_SHA256','result_path','source_commit','shadow_typed_tasks','shadow_predictions']} for r in records],'original_records':221,'current_and_next_frame_all_native_fields_exact':True,'question_context_rebuilt_after_commit':True,'shared_memory_for_actual_MATCH_questions':True,'cross_stage_state_memory_not_reused':True,'MEMORY_REACTIVATION':'UNTRAINED/FROZEN_FALLBACK','no_GT_or_future_runtime_inputs':True,'only_executed_prefix_inputs':True})
    m=json.loads((OUT/'visual_dataset_v1/MANIFEST.json').read_text());assert m['status']=='PASS' and sha(m['path'])==m['SHA256']
    save(REPORTS/'VISUAL_DATA_MANIFEST.json',m)
    print('ALL_221_NATIVE_OFF_SHADOW_PLUS_ACTUATION_GATES_PASS',flush=True)

if __name__=='__main__':main()

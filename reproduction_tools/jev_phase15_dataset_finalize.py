"""Qualify complete causal TRAIN data and actual persistent native replay."""
import collections
from jev_phase15_common import *


def main():
    protect();cases=[];counts=collections.Counter()
    for video in TRAIN:
        path=OUT/'commitment_dataset_v2'/f'video{video:02d}'/'RESULT.json'
        if not path.exists():
            print('PHASE15_DATASET_NOT_READY',video,flush=True);return
        result=read(path);assert result['status']=='PASS'
        assert sha(result['DATASET']['path'])==result['DATASET']['SHA256']
        assert result['exact_original_native_predictions']
        assert result['persistent_restore_inputs_logits_IDS_and_final_memory_exact']
        assert result['persistent_restore_payloads_exact']==128
        counts.update(result['counts']);cases.append(dict(result_manifest=ref(path),video=video,DATASET=result['DATASET'],
            records=result['records'],counts=result['counts'],duplicates_GT_labels_masked=result['duplicate_GT_offline_labels_masked']))
    source=binding(seed=20261009,dataset=cases,evaluator='actual frozen native TRAIN corpus and 512 restored camera payloads',
        scope='data/contract qualification only; no model quality claim')
    audit_correction=counts.get('audit_commit_kind_2',0)>0
    result=dict(status='PASS',binding=source,cases=cases,counts=dict(counts),
        TRAIN_ONLY=True,DEVELOPMENT_gradient_use=False,downloaded_GT_rewritten=False,
        WHO_multi_positive_preserved=counts.get('WHO_multi_positive_rows',0)>0,
        commitment_safe_training_support=counts.get('commit_kind_1',0)-counts.get('audit_commit_kind_1',0),
        commitment_safe_reserved_support=counts.get('audit_commit_kind_1',0),
        necessary_correction_training_support=counts.get('commit_kind_2',0)-counts.get('audit_commit_kind_2',0),
        necessary_correction_reserved_support=counts.get('audit_commit_kind_2',0),
        reserved_correction_assessment_eligible=audit_correction,
        certification='past observed GT coverage>=0.8, >=3 known observations; current duplicated GT identities masked as UNKNOWN',
        unknown_does_not_receive_wrong_person_certificate=True,
        label_support_is_not_tracking_GO=True)
    save(REPORTS/'COMMITMENT_LABEL_AUDIT.json',result)
    save(REPORTS/'UNKNOWN_RELIABILITY_AUDIT.json',dict(status='DATA_CERTIFIED_MODEL_NOT_YET_TRAINED',binding=source,
        purity_pure_options=counts.get('trust_pure'),purity_mixed_options=counts.get('trust_polluted'),
        mixed_current_compatibility='UNKNOWN; not automatically a wrong person',
        missing_history_uncertainty_labels='masked; absence of GT certification is not a reliable negative',
        uncertainty_certified_mixture=1,uncertainty_certified_pure=0,
        unknown_actions_legal=True,purity_accuracy=None,safety_accuracy=None,risk_coverage=None))
    prior=ROOT/'reports/JEV_PHASE13/NATIVE_PARITY.json'
    save(REPORTS/'NATIVE_STATE_CONTRACT.json',dict(status='PASS_FOR_IMPLEMENTED_NO_ALIAS_ADAPTER',binding=source,
        exact_full_TRAIN_original_predictions=True,actual_restore_payloads_bitwise=512,
        restored_inputs_logits_IDs_final_memory_exact=True,GTA_throw='PASS_IN_EACH_ACTUAL_TRAIN_RUN',
        snapshot_serialization='PASS_IMMEDIATE_IMMUTABLE_SERIALIZATION',native_solver_and_bank_source_unchanged=True,
        legacy_OFF_and_recycling_evidence=ref(prior),legacy_OFF_scope='historical measured contract reused; OFF/GTR source unchanged, new adapter only attaches in JEV_DIRECT',
        current_structural_tests=ref(REPORTS/'STRUCTURAL_TESTS.json'),
        fragment_alias='NOT_RUN_DISABLED_UNQUALIFIED_OPTIONAL',no_GT_or_future_in_new_state=True,
        learned_policy_continuation_correction_quality='PENDING_PILOT'))
    print('PHASE15_DATASET_QUALIFICATION_PASS',dict(counts),'reserved_correction',audit_correction,flush=True)


if __name__=='__main__':main()

"""Prevent positive aggregate scores from hiding a required real-case failure."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from write_worldobject_v4_report import qualify_state_variants,RETAINED_CASE_CHECKPOINT_SHA


def evidence():
    variants={key:{'original_legal_TRAIN_VAL_TrackEval':dict(HOTA=50+i,AssA=50+i,IDF1=60+i),
        'paired_against_C0':dict(temporal_corrections=10,temporal_regressions=2,
            same_person_corrections=5,same_person_regressions=1,
            different_person_corrections=3,different_person_regressions=1)}
        for i,key in enumerate(('C0','C1','C2','C3','C4'))}
    passed=dict(passes_targeted_mechanism=True,recovered_camera_keeps_ID=True,
        different_current_person_merged=False,ID_at704=17,ID_at705=17)
    report={'status':'COMPLETE','TRAIN_only':True,'separate_original_retained_epoch1_failure_case':{
        'checkpoint':{'checkpoint_sha256':RETAINED_CASE_CHECKPOINT_SHA},
        'assertions':{'C0_original_158_to_162_found':True,'variants':{'C3':deepcopy(passed),'C4':deepcopy(passed)}}}}
    return variants,report


def qualify(variants,report,**kwargs):
    return qualify_state_variants(variants,report,cpu_passed=kwargs.get('cpu_passed',True),
        audits_complete=kwargs.get('audits_complete',True))


def test_missing_case_does_not_qualify_even_when_all_train_scores_improve():
    v,_=evidence();r=qualify(v,{})
    assert r['TRAIN_metric_eligible_state_variants']==['C1','C2','C3','C4']
    assert not r['full_training_qualified'] and not r['C5_live_quality_gate_passed']
    assert r['TRAIN_selected_state_variant'] is None


@pytest.mark.parametrize('field,value',[
    ('passes_targeted_mechanism',False),('recovered_camera_keeps_ID',False),
    ('different_current_person_merged',True),('ID_at705',18),('ID_at704',None),('ID_at704',True)])
def test_failed_retained_c4_cannot_trigger_c5_live(field,value):
    v,e=evidence();e['separate_original_retained_epoch1_failure_case']['assertions']['variants']['C4'][field]=value
    r=qualify(v,e)
    assert r['TRAIN_metric_selected_state_variant']=='C4'
    assert r['TRAIN_selected_state_variant']=='C3'
    assert r['retained_original_case_by_variant']['C4']=='FAILED'
    assert not r['C5_live_quality_gate_passed']


@pytest.mark.parametrize('change',['wrong_checkpoint','original_case_not_reproduced','not_train','incomplete'])
def test_unverified_original_case_provenance_cannot_qualify(change):
    v,e=evidence();case=e['separate_original_retained_epoch1_failure_case']
    if change=='wrong_checkpoint':case['checkpoint']['checkpoint_sha256']='0'*64
    elif change=='original_case_not_reproduced':case['assertions']['C0_original_158_to_162_found']=False
    elif change=='not_train':e['TRAIN_only']=False
    else:e['status']='INCOMPLETE'
    r=qualify(v,e)
    assert not r['retained_original_case_provenance_verified']
    assert not r['full_training_qualified'] and not r['C5_live_quality_gate_passed']


def test_true_case_pass_still_requires_native_gain_and_no_net_merge_regression():
    v,e=evidence();v['C4']['paired_against_C0']['different_person_regressions']=4
    r=qualify(v,e)
    assert r['retained_original_case_by_variant']['C4']=='PASSED'
    assert 'C4' not in r['TRAIN_metric_eligible_state_variants']
    assert not r['C5_live_quality_gate_passed']
    v['C3']['original_legal_TRAIN_VAL_TrackEval']['HOTA']=49
    assert not qualify(v,e)['full_training_qualified']


def test_case_pass_and_metric_gain_require_completed_audits_for_next_training():
    v,e=evidence();r=qualify(v,e)
    assert r['full_training_qualified'] and r['TRAIN_selected_state_variant']=='C4'
    assert r['C5_live_quality_gate_passed']
    assert not qualify(v,e,audits_complete=False)['full_training_qualified']
    cpu_failed=qualify(v,e,cpu_passed=False)
    assert not cpu_failed['full_training_qualified'] and not cpu_failed['C5_live_quality_gate_passed']

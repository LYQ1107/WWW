import numpy as np
import pytest
from scripts.gmt_2d_control import Control,Settings
from worldmot.models.person_detector import PersonDetection


def det(v,i,x,feature=(1.,0.)):
    d=PersonDetection(v,i,(x,0.,10.,20.),.9)
    d.detector_roi_feature=np.asarray(feature);return d


def test_iou_control_preserves_current_candidates_and_does_not_join_views():
    model=Control(Settings())
    first=model.step({1:[det(1,0,0)],2:[det(2,0,0)]},1,8)
    second=model.step({1:[det(1,0,1)],2:[det(2,0,1)]},2,8)
    assert first['tracks_2d_by_camera']['1'][0]['global_id']!=first['tracks_2d_by_camera']['2'][0]['global_id']
    assert second['tracks_2d_by_camera']['1'][0]['global_id']==first['tracks_2d_by_camera']['1'][0]['global_id']
    assert second['output_count']==second['raw_candidate_count']==2


def test_appearance_control_joins_synchronous_births_from_independent_roi_only():
    model=Control(Settings(appearance=True))
    row=model.step({1:[det(1,0,0)],2:[det(2,0,300)]},1,8)
    assert row['tracks_2d_by_camera']['1'][0]['global_id']==row['tracks_2d_by_camera']['2'][0]['global_id']
    assert row['decisions'][0]['kind']=='CURRENT_CROSS_VIEW_ROI_MATCH'


def test_no_forced_match_or_duplicate_same_camera_id():
    model=Control(Settings(appearance=True))
    model.step({1:[det(1,0,0)],2:[]},1,8)
    row=model.step({1:[det(1,0,1),det(1,1,100)],2:[det(2,0,0,(0.,1.))]},2,8)
    ids=[x['global_id'] for x in row['tracks_2d_by_camera']['1']]
    assert len(set(ids))==2
    assert row['tracks_2d_by_camera']['2'][0]['global_id'] not in ids


def test_lost_emits_no_prediction_and_scene_reset_is_causal():
    model=Control(Settings(lost_retention=2))
    model.step({1:[det(1,0,0)]},1,8)
    assert model.step({1:[]},2,8)['tracks_2d_by_camera']=={'1':[]}
    row=model.step({1:[det(1,0,0)]},5,8)
    assert row['tracks_2d_by_camera']['1'][0]['global_id']==2
    with pytest.raises(ValueError):model.step({1:[]},5,8)
    assert model.step({1:[det(1,0,0)]},1,9)['tracks_2d_by_camera']['1'][0]['global_id']==1


def test_appearance_cannot_silently_fallback_to_fake_or_missing_features():
    d=PersonDetection(1,0,(0,0,10,20),.9)
    with pytest.raises(ValueError):Control(Settings(appearance=True)).step({1:[d]},1,8)

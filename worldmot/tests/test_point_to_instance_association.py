import numpy as np
from worldmot.models.person_detector import PersonDetection
from worldmot.tracking.point_to_instance_association import (PointAssociationSettings,point_support,
    exclusive_assignment,bidirectional_matches)


def test_invalid_or_low_support_points_cannot_vote_for_other_person():
    settings=PointAssociationSettings()
    det=PersonDetection(2,0,(10,10,20,20),.9)
    xy=np.array([[15,15]]*16)
    ev=point_support(xy,np.ones(16),np.ones(16),np.array([True]+[False]*15),
                     np.ones(16),det,settings,1,2,1)
    assert ev['inside_box_count']==1 and ev['association_score']==0
    assert ev['point_support_ratio']==1/16
    ev=point_support(xy,np.zeros(16),np.ones(16),np.ones(16,bool),np.ones(16),det,settings,1,2,0)
    assert ev['valid_point_count']==0 and ev['rejection_reason']=='INSUFFICIENT_VALID_POINTS'


def test_global_exclusivity_with_null_and_bidirectional_requirement():
    matches=exclusive_assignment([[.8,.7],[.9,.1],[.01,.01]],.2)
    assert matches==[(0,1),(1,0)]
    assert exclusive_assignment(np.empty((0,4)),.2)==[]
    assert exclusive_assignment([[np.nan,.1]],.2)==[]
    assert bidirectional_matches([[.9]],[[.1]],.2)==[]
    assert bidirectional_matches([[.9]],[[.8]],.2)[0][:2]==(0,0)


def test_no_overlap_returns_null_instead_of_nearest_person():
    det=PersonDetection(2,0,(100,100,20,20),.9)
    ev=point_support(np.full((16,2),15.),np.ones(16),np.ones(16),np.ones(16,bool),
                     np.ones(16),det,PointAssociationSettings(),1,2,0)
    assert ev['association_score']==0 and ev['inside_box_count']==0

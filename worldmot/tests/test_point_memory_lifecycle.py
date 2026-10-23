import numpy as np
from point_fakes import PointOracle,scene_detections,images
from worldmot.tracking.point_identity_memory import PointIdentityMemory
from worldmot.tracking.point_to_instance_association import PointAssociationSettings
from worldmot.tracking.person_point_sampler import PersonPointSampler
from worldmot.models.person_detector import PersonDetection


def test_short_loss_uses_actual_historical_anchor_and_no_detection_emits_nothing():
    oracle=PointOracle();memory=PointIdentityMemory(oracle)
    memory.step(images(),scene_detections(),1,8)
    gap=memory.step(images(),{1:[],2:[]},2,8)
    assert gap['output_count']==0 and memory.tracks[1].status=='LOST'
    restored=memory.step(images(),scene_detections(3),3,8)
    assert restored['tracks_2d_by_camera']['1'][0]['global_id']==1
    assert any(k[1]==1 for call,_,_ in oracle.calls for k in call)


def test_reference_window_expiration_births_new_id_and_scene_resets_memory():
    memory=PointIdentityMemory(PointOracle(),PointAssociationSettings(maximum_reference_gap=1,lost_retention=2))
    memory.step(images(),scene_detections(),1,8)
    memory.step(images(),{1:[],2:[]},2,8)
    row=memory.step(images(),scene_detections(3),3,8)
    assert row['tracks_2d_by_camera']['1'][0]['global_id']!=1
    reset=memory.step(images(),scene_detections(),1,9)
    assert reset['tracks_2d_by_camera']['1'][0]['global_id']==1 and len(memory.tracks)==1


def test_deterministic_sampler_uses_mask_and_handles_partial_box_and_empty_mask():
    image=np.zeros((40,60,3),np.uint8)
    mask=np.zeros((40,60),bool);mask[10:30,5:20]=True
    det=PersonDetection(1,0,(-5,0,40,40),.9,mask=mask)
    sampler=PersonPointSampler(16)
    first=sampler.sample(image,det,1,9);second=sampler.sample(image,det,1,9)
    assert np.array_equal(first.xy,second.xy)
    assert len(first.xy)==16 and first.global_id==9
    pix=np.rint(first.xy).astype(int)
    assert mask[pix[:,1],pix[:,0]].all()
    det.mask=np.zeros_like(mask)
    assert sampler.sample(image,det,1).xy.shape==(0,2)

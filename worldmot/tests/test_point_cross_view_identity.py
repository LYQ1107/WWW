from point_fakes import PointOracle,scene_detections,images
from worldmot.tracking.point_identity_memory import PointIdentityMemory


def test_newborn_cross_view_merge_precedes_id_allocation_and_time_is_persistent():
    memory=PointIdentityMemory(PointOracle())
    first=memory.step(images(),scene_detections(),1,8)
    assert first['tracks_2d_by_camera']['1'][0]['global_id']==first['tracks_2d_by_camera']['2'][0]['global_id']==1
    second=memory.step(images(),scene_detections(2),2,8)
    assert {r['global_id'] for ds in second['tracks_2d_by_camera'].values() for r in ds}=={1}
    assert len(memory.tracks)==1


def test_temporal_only_does_not_claim_cross_camera_identity_and_low_confidence_keeps_boxes():
    temporal=PointIdentityMemory(PointOracle(),cross_view=False)
    first=temporal.step(images(),scene_detections(),1,8)
    assert first['tracks_2d_by_camera']['1'][0]['global_id']!=first['tracks_2d_by_camera']['2'][0]['global_id']
    uncertain=PointIdentityMemory(PointOracle(confidence=0.))
    for t in [1,2]:
        row=uncertain.step(images(),scene_detections(t),t,8)
        assert row['output_count']==row['raw_candidate_count']==2
        assert all(r['visible'] and r['evaluation_eligible'] for ds in row['tracks_2d_by_camera'].values() for r in ds)


def test_two_detections_in_one_camera_cannot_share_identity():
    dets=scene_detections()
    from worldmot.models.person_detector import PersonDetection
    dets[1].append(PersonDetection(1,1,(11,10,20,40),.8))
    memory=PointIdentityMemory(PointOracle())
    row=memory.step(images(),dets,1,8)
    ids=[r['global_id'] for r in row['tracks_2d_by_camera']['1']]
    assert len(ids)==len(set(ids))==2

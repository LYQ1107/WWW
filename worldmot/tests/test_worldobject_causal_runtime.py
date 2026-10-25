import json
from pathlib import Path
import numpy as np
import pytest
from worldobject_synthetic import obs,core
from worldmot.data.worldobject_observations import live_observations,replay_observations
from worldmot.geometry.world_frame_contract import WorldFrameTransform
from worldmot.geometry.world_alignment import Sim3
from worldmot.models.geometry_interface import GeometryFrame
from worldmot.geometry.camera_model import CameraModel
from worldmot.models.person_detector import PersonDetection
from worldmot.runtime.online_world_object_tracker import OnlineWorldObjectTracker
from worldmot.runtime.experiment_authorization import require_experiment_authorization
from worldmot.data.visiontrack import VisionTrack


def synthetic_frontend(timestamp=0,status="OK"):
    image=np.zeros((8,8,3),np.uint8)
    yy,xx=np.mgrid[:8,:8]
    points=np.stack([xx*.01,yy*.01,np.ones((8,8))*3],axis=-1)
    camera=CameraModel(np.eye(3),np.eye(4),(8,8),(8,8),np.eye(3))
    frame=GeometryFrame(timestamp,{1:image,2:image},{1:points,2:points},{1:np.ones((8,8))*2,2:np.ones((8,8))*2},
                        {},np.empty((0,0)),{1:camera,2:camera},image_to_model={1:np.eye(3),2:np.eye(3)},status=status)
    detections={}
    for v in (1,2):
        d=PersonDetection(v,0,(0,0,8,8),.9);d.detector_roi_feature=np.array([1.,0.,0.])
        detections[v]=[d]
    return frame,detections


def test_live_and_replay_adapters_produce_identical_core_ids_boxes_and_state():
    frame,detections=synthetic_frontend()
    observations=live_observations(detections,frame,WorldFrameTransform(Sim3(),0,True))
    rows=[]
    for o in observations:
        s=o.support
        rows.append({"view":o.camera_id,"detection_id":o.detection_id,"bbox":o.bbox_xywh,"score":o.score,
            "roi_feature":o.roi_feature,"feature_valid":True,"center":s.center,"cov":s.covariance,
            "box":s.box_points,"foreground":s.foreground_points,"quality":s.quality,
            "foreground_fraction":s.foreground_fraction,"geometry_valid":o.geometry_valid,"geometry_status":o.geometry_status})
    replay=replay_observations({"timestamp":0,"rows":rows})
    live_core=core();replay_core=core()
    live=OnlineWorldObjectTracker(None,None,live_core.config,core=live_core)
    live.reset(1)
    a=live.step_observations(observations,0);b=replay_core.step(replay,0)
    assert a==b
    for gid_,state in live_core.bank.objects.items():
        np.testing.assert_array_equal(state.identity_embedding,replay_core.bank.objects[gid_].identity_embedding)


def test_alignment_failure_in_live_frontend_does_not_reuse_transform():
    frame,detections=synthetic_frontend(1,"INVALID_WORLD_GEOMETRY")
    class Detector:
        def infer(self,images):return detections
    class Geometry:
        def reconstruct(self,images,timestamp):return frame
    c=core();tracker=OnlineWorldObjectTracker(Detector(),Geometry(),c.config,core=c)
    tracker.reset(1);c.step([obs(1),obs(2)],0)
    old=c.bank.objects[1].center.copy()
    r=tracker.step(frame.images,1)
    assert not r["alignment"]["success"] and not r["alignment"]["historical_transform_reused"]
    np.testing.assert_array_equal(old,c.bank.objects[1].center)
    assert sum(map(len,r["tracks_2d_by_camera"].values()))==2


@pytest.mark.parametrize("mutator",["future","duplicate","timestamp","three_cameras"])
def test_invalid_time_and_duplicate_inputs_fail_before_state_mutation(mutator):
    c=core();c.step([obs()],0)
    if mutator=="future":args=([obs(timestamp=2)],1)
    elif mutator=="duplicate":args=([obs(timestamp=1),obs(timestamp=1)],1)
    elif mutator=="timestamp":args=([obs()],0)
    else:args=([obs(1,timestamp=1),obs(2,timestamp=1),obs(3,timestamp=1)],1)
    with pytest.raises(ValueError):c.step(*args)
    assert c.bank.last_timestamp==0 and len(c.bank.objects)==1


def test_scene_reset_clears_history_and_camera_order_is_invariant():
    a,b=core(),core()
    observations=[obs(2),obs(1)]
    assert a.step(observations,0)==b.step(list(reversed(observations)),0)
    a.step([obs(timestamp=1)],1,scene_id=2)
    assert a.scene_id==2 and len(a.bank.objects)==1 and a.bank.next_id==2


def test_metadata_only_dataset_isolates_gt_and_rejects_duplicate_image_keys(tmp_path):
    directory=tmp_path/"annotations";directory.mkdir()
    data={"videos":[{"id":1,"file_name":"scene1"}],"images":[
        {"id":1,"video_id":1,"view_id":1,"frame_id":0,"file_name":"1.jpg"}],
        "annotations":[{"image_id":1,"instance_id":99}]}
    (directory/"test.json").write_text(json.dumps(data))
    dataset=VisionTrack(tmp_path,split="test",load_annotations=False)
    assert dataset._annotations=={}
    with pytest.raises(RuntimeError,match="forbids GT"):
        dataset.ground_truth(1)
    data["images"].append(data["images"][0])
    (directory/"test.json").write_text(json.dumps(data))
    with pytest.raises(ValueError,match="DUPLICATE"):
        VisionTrack(tmp_path,split="test",load_annotations=False)


def test_missing_approval_blocks_before_any_real_io(monkeypatch):
    def no_io(*args,**kwargs):raise AssertionError("must not open a dataset or weight")
    monkeypatch.setattr(Path,"open",no_io)
    with pytest.raises(PermissionError,match="WAITING_USER_APPROVAL"):
        require_experiment_authorization(None,"train","train","FULL_WORLD_OBJECT_V3")


def test_old_checkpoint_gate_is_rejected_without_loading_weights(tmp_path):
    from worldmot.models.world_object_checkpoint import load_identity_checkpoint
    path=tmp_path/"legacy.json";path.write_text(json.dumps({"schema":"worldobject_w4_fulltrain_gate_v1"}))
    with pytest.raises(ValueError,match="INCOMPATIBLE"):
        load_identity_checkpoint(path,core().config)

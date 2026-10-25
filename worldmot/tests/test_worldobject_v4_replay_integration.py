"""Synthetic end-to-end writer/causal replay preflight, NOT model evaluation."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np


def test_all_registered_replay_paths_write_and_preserve_legacy_without_NAS(tmp_path,monkeypatch):
    analysis=Path(__file__).resolve().parents[1]/'analysis'
    monkeypatch.syspath_prepend(str(analysis))
    module=importlib.import_module('run_worldobject_v4_train_replay')
    import torch
    from worldmot.models.world_object_update_head import WorldObjectUpdateHead
    from worldmot.models.person_detector import PersonDetection
    feature=np.zeros(1024,np.float32);feature[0]=1
    def detections():
        result={1:[],2:[]}
        for camera in result:
            d=PersonDetection(camera,0,(0,0,10,20),.9);d.detector_roi_feature=feature.copy();result[camera]=[d]
        return result
    class Raw:
        feature_dim=1024;lock={'split':'train'}
        def __init__(self,*args,**kwargs):pass
        def get(self,*args,**kwargs):return detections()
    class Dataset:
        def __init__(self,*args,**kwargs):assert kwargs['split']=='train'
        def synchronized_pairs(self,scene):
            for t in (1,2):yield SimpleNamespace(frame_id=t,images={1:SimpleNamespace(image_id=t),2:SimpleNamespace(image_id=t+2)})
        def ground_truth_pair(self,pair):
            return {v:[{'instance_id':7,'bbox':[0,0,10,20],'category_id':1}] for v in (1,2)}
    class Cache:
        def __init__(self,*args,**kwargs):pass
        def frames(self):
            for t in (1,2):
                rows=[]
                for camera in (1,2):
                    center=np.array([0.,0,3.]);points=center+np.arange(36).reshape(12,3)*.001
                    rows.append(dict(view=camera,detection_id=0,bbox=[0,0,10,20],score=.9,roi_feature=feature,
                        feature_valid=True,center=center,cov=np.eye(3)*.01,box=points,foreground=points,
                        quality=1.,foreground_fraction=1.,support_ambiguity='DEPTH_CLUSTERED',geometry_valid=True,
                        geometry_status='GEOMETRY_VALID',point_support_available=True,cross_view_fusion_eligible=True,
                        world_state_update_eligible=True))
                yield dict(timestamp=t,rows=rows,record={'alignment':{'success':True}},source={'schema':'SYNTHETIC_UNIT_ONLY'})
    def load_head(*args,**kwargs):
        torch.manual_seed(42);head=WorldObjectUpdateHead(1024,8).eval().requires_grad_(False)
        return head,{'checkpoint_sha256':'SYNTHETIC_TEST_WEIGHTS_ONLY','optimizer_used':False}
    monkeypatch.setattr('worldmot.models.gmt_detection_cache.ImmutableGMTDetectionCache',Raw)
    monkeypatch.setattr('worldmot.data.visiontrack.VisionTrack',Dataset)
    monkeypatch.setattr('worldmot.data.worldobject_cache.WorldObjectCache',Cache)
    monkeypatch.setattr('worldmot.models.world_object_v4_checkpoint.load_frozen_r2_head',load_head)
    monkeypatch.setattr(sys,'argv',['unit','--r2-root',str(tmp_path/'unused_R2'),'--output',str(tmp_path/'out'),
                                 '--scene','4','--limit','2','--with-single-step'])
    module.main()
    report=json.loads((tmp_path/'out/replay_report.json').read_text())
    assert report['status']=='LIMITED_PREFLIGHT' and report['C0_R2_exact_frames']==2
    assert len(report['closed_loop'])==15 and len(report['factorial_single_factor_edges'])==12
    assert report['B1_fixed_C0_past']['same_step_G3_assignment_invariance_checked']
    assert report['checkpoint']['checkpoint_sha256']=='SYNTHETIC_TEST_WEIGHTS_ONLY'
    assert report['TEST_read'] is False and report['optimizer_used'] is False
    for variant in report['closed_loop']:
        rows=[json.loads(x) for x in (tmp_path/'out'/variant/'scene_04/tracks.jsonl').read_text().splitlines()]
        assert len(rows)==2 and all(sum(map(len,x['tracks_2d_by_camera'].values()))==2 for x in rows)

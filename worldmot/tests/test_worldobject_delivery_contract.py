"""Synthetic end-to-end cache contracts and fail-closed entrypoint boundaries."""
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from worldobject_synthetic import obs,core
from test_worldobject_memory import evidence
from test_worldobject_causal_runtime import synthetic_frontend
from test_gmt_detection_cache import fixture_cache

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"


def module(name):
    spec=importlib.util.spec_from_file_location("synthetic_"+name,SCRIPTS/(name+".py"))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def test_v2_producer_reader_feature_and_camera_contract(tmp_path,monkeypatch):
    from worldmot.data.worldobject_cache import WorldObjectCache,V2
    rawdir=tmp_path/"raw";rawdir.mkdir()
    raw,_,_=fixture_cache(rawdir)
    frame,_=synthetic_frontend(timestamp=1)
    class Omega:
        def reconstruct(self,images,timestamp):return frame
        def diagnostics(self):return {"checkpoint_sha256":"synthetic_only"}
    class Dataset:
        def synchronized_pairs(self,scene):
            return iter([SimpleNamespace(frame_id=1,images={1:SimpleNamespace(image_id=1),2:SimpleNamespace(image_id=2)})])
    producer=module("cache_worldobject_support")
    monkeypatch.setattr(producer,"read_images",lambda ds,pair:frame.images)
    producer.cache_scene(Dataset(),8,tmp_path/"support",Omega(),raw,"train",1)
    cache=WorldObjectCache(tmp_path/"support",8,raw_cache=raw,require_geometry=True)
    result=list(cache.frames())
    assert cache.schema==V2 and len(result)==1 and len(result[0]["rows"])==2
    assert len(result[0]["rows"][0]["roi_feature"])==8
    assert "camera_models" in result[0]["record"]
    payload=cache.path/"shard_0000.npz"
    with np.load(payload,allow_pickle=False) as arrays:
        assert "identity" not in arrays and arrays["roi_features"].shape==(2,8)
    side=payload.with_suffix(".json")
    side.write_text(side.read_text()+" ")
    with pytest.raises(RuntimeError,match="SIDECAR_SHA"):
        list(cache.frames())


@pytest.mark.parametrize("name,arguments",[
    ("cache_worldobject_support",["--output","unused","--raw-cache","unused","--scenes","1"]),
    ("train_world_object",["--cache","unused","--raw-cache","unused","--dataset-root","unused","--output","unused","--scope-lock","unused"]),
    ("run_world_object_fullvideo",["--cache","unused","--raw-cache","unused","--output","unused","--scope-lock","unused"]),
    ("run_world_object_live",["--dataset-root","unused","--raw-cache","unused","--output","unused","--scope-lock","unused"]),
    ("freeze_worldobject_scope",["--cache-train","unused","--raw-cache-train","unused","--output","unused"]),
    ("evaluate_fullvideo",["--split","test","--audit","unused","--predictions","unused","--output","unused","--method-id","FULL_WORLD_OBJECT_V3"]),
    ("orchestrate_worldobject_six_gpu",["--gpu-ids","0","--phase","all","--root","unused"]),
    ("evaluate_world_object_formation",["--cache","unused","--predictions","unused","--dataset-root","unused","--output","unused","--split","test","--method-id","FULL_WORLD_OBJECT_V3"]),
    ("run_world_object_legacy_control",["--cache","unused","--raw-cache","unused","--shared-scope-lock","unused","--output","unused","--split","train","--feature-mode","restored"]),
])
def test_real_entrypoints_deny_without_approval_before_opening_resources(name,arguments,monkeypatch):
    monkeypatch.setattr(sys,"argv",[name]+arguments)
    with pytest.raises(PermissionError,match="WAITING_USER_APPROVAL"):
        runpy.run_path(str(SCRIPTS/(name+".py")),run_name="__main__")


def test_bad_model_evidence_cannot_partially_mutate_history():
    c=core();c.step([obs()],0)
    ev=evidence(1,1);ev["memory_delta"][0,0,0]=np.nan
    with pytest.raises(ValueError,match="INVALID_LEARNED_EVIDENCE"):
        c.step([obs(timestamp=1)],1,evidence=ev)
    assert c.bank.last_timestamp==0 and c.bank.objects[1].missed==0


def test_teacher_decision_is_unavailable_to_inference():
    c=core()
    with pytest.raises(ValueError,match="TEACHER_DECISION_FORBIDDEN"):
        c.step([],0,training_decision=object())


def test_model_pair_limit_has_explicit_causal_fallback():
    from worldmot.models.world_object_update_head import WorldObjectUpdateHead
    c=core(max_model_pairs=1)
    c.step([obs(1,0),obs(1,1,feature=(0,1,0))],0)
    c.model=WorldObjectUpdateHead(3,16)
    result=c.step([obs(timestamp=1)],1)
    assert result["bank"]["evidence_status"]=="LEARNED_PAIR_LIMIT_RULE_EVIDENCE_FALLBACK"
    assert len(result["tracks_2d_by_camera"]["1"])==1


def test_assignment_logits_change_the_selected_historical_id():
    c=core()
    c.step([obs(1,0),obs(1,1,center=(.02,0,3))],0)
    ev=evidence(2,2)
    ev["assignment_logits"][:]=[[-20,20],[20,-20]]
    result=c.step([obs(1,0,timestamp=1),obs(1,1,center=(.02,0,3),timestamp=1)],1,evidence=ev)
    assert [row["global_id"] for row in result["tracks_2d_by_camera"]["1"]]==[2,1]


def test_stale_world_transform_is_rejected_even_if_marked_valid():
    from worldmot.data.worldobject_observations import live_observations
    from worldmot.geometry.world_frame_contract import WorldFrameTransform
    from worldmot.geometry.world_alignment import Sim3
    frame,detections=synthetic_frontend(timestamp=2)
    with pytest.raises(ValueError,match="STALE_WORLD_FRAME_TRANSFORM"):
        live_observations(detections,frame,WorldFrameTransform(Sim3(),1,True))


def test_detection_source_camera_id_must_match_mapping():
    from worldmot.data.worldobject_observations import live_observations
    from worldmot.geometry.world_frame_contract import WorldFrameTransform
    from worldmot.geometry.world_alignment import Sim3
    frame,detections=synthetic_frontend()
    detections[1][0].view_id=2
    with pytest.raises(ValueError,match="CAMERA_SOURCE_MISMATCH"):
        live_observations(detections,frame,WorldFrameTransform(Sim3(),0,True))


def test_posthoc_diagnostics_do_not_mutate_predictions():
    from worldmot.evaluation.world_object_diagnostics import identity_diagnostics
    observations=[obs(1),obs(2)]
    mapping={(1,0):17,(2,0):17}
    counts,failures=identity_diagnostics(observations,["A","B"],mapping,{})
    assert counts["wrong_world_object_merges"]==1 and failures
    assert mapping=={(1,0):17,(2,0):17}


def test_timing_summary_uses_wall_latency_and_serial_throughput():
    summary=module("summarize_worldobject_fps").timing_statistics([1.,2.,3.])
    assert summary["p50_s"]==2. and summary["serial_throughput_pairs_per_second"]==.5


def test_complete_scene_cache_is_reused_without_rerunning_geometry(tmp_path,monkeypatch):
    from test_gmt_detection_cache import fixture_cache
    rawdir=tmp_path/"raw";rawdir.mkdir();raw,_,_=fixture_cache(rawdir)
    frame,_=synthetic_frontend(timestamp=1)
    class Omega:
        calls=0
        def reconstruct(self,images,timestamp):self.calls+=1;return frame
        def diagnostics(self):return {"checkpoint_sha256":"synthetic_only"}
    class Dataset:
        def synchronized_pairs(self,scene):return iter([SimpleNamespace(frame_id=1,images={1:SimpleNamespace(image_id=1),2:SimpleNamespace(image_id=2)})])
    producer=module("cache_worldobject_support");monkeypatch.setattr(producer,"read_images",lambda ds,pair:frame.images)
    omega=Omega();producer.cache_scene(Dataset(),8,tmp_path/"output",omega,raw,"train",1)
    producer.cache_scene(Dataset(),8,tmp_path/"output",omega,raw,"train",1)
    assert omega.calls==1


def test_partial_scene_is_recoverably_quarantined_before_restart(tmp_path,monkeypatch):
    from test_gmt_detection_cache import fixture_cache
    rawdir=tmp_path/"raw";rawdir.mkdir();raw,_,_=fixture_cache(rawdir)
    frame,_=synthetic_frontend(timestamp=1)
    class Omega:
        def reconstruct(self,images,timestamp):return frame
        def diagnostics(self):return {"checkpoint_sha256":"synthetic_only"}
    class Dataset:
        def synchronized_pairs(self,scene):return iter([SimpleNamespace(frame_id=1,images={1:SimpleNamespace(image_id=1),2:SimpleNamespace(image_id=2)})])
    destination=tmp_path/"output"/"scene_08";destination.mkdir(parents=True)
    (destination/"partial.tmp").write_text("synthetic partial shard, retained")
    producer=module("cache_worldobject_support");monkeypatch.setattr(producer,"read_images",lambda ds,pair:frame.images)
    producer.cache_scene(Dataset(),8,tmp_path/"output",Omega(),raw,"train",1)
    preserved=list((tmp_path/"output"/"failed_scenes").glob("scene_08.*"))
    assert len(preserved)==1 and (preserved[0]/"partial.tmp").read_text()=="synthetic partial shard, retained"
    assert (destination/"manifest.json").is_file()


def test_parallel_scene_claims_are_unique(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    worker=module("worldobject_v3_cache_worker")
    for scene in range(1,7):
        (tmp_path/f"scene_{scene:02d}.pending").write_text(json.dumps({"split":"train","scene":scene,"frames":100}))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda _:worker.claim_scene(tmp_path,"train"),range(16)))
    scenes=[task["scene"] for _,task in results if task is not None]
    assert sorted(scenes)==list(range(1,7)) and len(list(tmp_path.glob("*.running")))==6


def test_native_matlab_hash_scope_excludes_unused_detector_tree(tmp_path):
    from worldmot.evaluation.official_fullvideo import native_tracking_sources
    utils=tmp_path/"utils";utils.mkdir()
    (tmp_path/"evaluateTracking.m").write_text("% synthetic tracking function")
    (utils/"IDmeasures.m").write_text("% synthetic identity function")
    detector=utils/"external"/"dollar"/"detector";detector.mkdir(parents=True)
    (detector/"unused.m").write_text("% excluded synthetic detector")
    paths=native_tracking_sources(tmp_path)
    assert len(paths)==2 and all("dollar" not in str(path) for path in paths)


def test_source_commit_reader_handles_worktree_and_packed_refs_without_fork(tmp_path):
    from worldmot.runtime.experiment_authorization import read_git_head
    repository=tmp_path/"checkout";repository.mkdir()
    common=tmp_path/"common";common.mkdir()
    worktree=common/"worktrees"/"test";worktree.mkdir(parents=True)
    (repository/".git").write_text("gitdir: "+str(worktree))
    (worktree/"HEAD").write_text("ref: refs/heads/research/test\n")
    (worktree/"commondir").write_text("../..\n")
    (common/"packed-refs").write_text("# pack-refs\n"+"a"*40+" refs/heads/research/test\n")
    assert read_git_head(repository)=="a"*40


def test_legacy_feature_control_is_explicit_and_not_imported_by_primary_core():
    control=module("run_world_object_legacy_control")
    observations=[obs(1),obs(2)]
    restored,_=control.control_hypotheses(observations,"restored")
    disabled,_=control.control_hypotheses(observations,"legacy_disabled")
    np.testing.assert_allclose(restored[0].feature,[1,0,0])
    np.testing.assert_array_equal(disabled[0].feature,[0])
    from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
    import inspect
    assert "legacy_worldobject" not in inspect.getsource(WorldObjectTrackingCore)


def test_report_only_commit_changes_do_not_invalidate_identical_code_scope(tmp_path):
    from worldmot.data.worldobject_scope import load_scope_lock,SCOPE_SCHEMA
    from worldmot.runtime.experiment_authorization import source_fingerprint
    path=tmp_path/"scope.json"
    scope={"schema":SCOPE_SCHEMA,"source_commit":"a"*40,"source_sha256":source_fingerprint(),
           "method_id":"FULL_WORLD_OBJECT_V3","splits":{"test":{"scene_count":22,"timestamps":29019}}}
    path.write_text(json.dumps(scope))
    assert load_scope_lock(path,"test","FULL_WORLD_OBJECT_V3")["source_commit"]=="a"*40
    scope["source_sha256"]="changed";path.write_text(json.dumps(scope))
    with pytest.raises(ValueError,match="FROZEN_SCOPE_MISMATCH"):
        load_scope_lock(path,"test","FULL_WORLD_OBJECT_V3")

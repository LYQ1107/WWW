import json
import numpy as np
import pytest
from worldobject_synthetic import obs,core
from test_gmt_detection_cache import fixture_cache
from worldmot.data.worldobject_feature_join import FrozenGMTFeatureJoin
from worldmot.data.worldobject_cache import WorldObjectCache,V1
from worldmot.models.world_object_observation import FeatureUnavailable,normalized_feature,fuse_identity_features
from worldmot.data.worldobject_observations import replay_observations
from worldmot.models.person_detector import sha256_file


def test_roi_feature_survives_observation_assignment_and_bank():
    c=core();o=obs(feature=(2,1,0))
    c.step([o],0)
    q=c.bank.objects[1]
    np.testing.assert_allclose(q.identity_embedding,normalized_feature([2,1,0]))
    assert q.identity_embedding.shape==(3,)
    assert c.bank.last_hypotheses[0].feature_valid


def test_invisible_view_does_not_write_zero_memory():
    visible=obs(feature=(1,0,0));invisible=obs(2,feature=None);invisible.visible=False
    f,_=fuse_identity_features([visible,invisible],3)
    np.testing.assert_array_equal(f,[1,0,0])


def test_missing_features_fail_except_explicit_geometry_only():
    with pytest.raises(FeatureUnavailable,match="FEATURE_UNAVAILABLE"):
        core().step([obs(feature=None)],0)
    with pytest.raises(FeatureUnavailable):
        normalized_feature(np.zeros(3))
    with pytest.raises(FeatureUnavailable,match="DIMENSION"):
        normalized_feature(np.ones(7),3)
    c=core("GEOMETRY_ONLY_ABLATION_V3")
    c.step([obs(feature=None)],0)
    assert c.bank.objects[1].identity_embedding is None


def joined_rows(cache):
    detections=cache.get(8,1)
    return [{"view":v,"detection_id":d.detection_id,"bbox":d.bbox_xywh,"score":d.score}
            for v,ds in detections.items() for d in ds]


def test_readonly_feature_join_checks_dimension_count_bbox_score_and_sha(tmp_path):
    cache,_,path=fixture_cache(tmp_path)
    before=sha256_file(path)
    join=FrozenGMTFeatureJoin(cache,{"detector_cache":cache.manifest()})
    rows=joined_rows(cache);record={"image_ids":{"1":1,"2":2}}
    out=join.join(8,1,rows,record)
    assert len(out[0]["roi_feature"])==8 and sha256_file(path)==before
    with pytest.raises(ValueError,match="COUNT_OR_ONE_TO_ONE"):
        join.join(8,1,rows[:1],record)
    with pytest.raises(ValueError,match="BBOX"):
        join.join(8,1,[{**rows[0],"bbox":[9,9,9,9]},rows[1]],record)
    with pytest.raises(ValueError,match="SCORE"):
        join.join(8,1,[{**rows[0],"score":.1},rows[1]],record)
    with pytest.raises(ValueError,match="SHARD_SHA"):
        join.join(8,1,rows,{**record,"raw_candidate_shard_sha256":"wrong"})
    with pytest.raises(ValueError,match="LOCK_SHA"):
        FrozenGMTFeatureJoin(cache,{"detector_cache":{"cache_lock_sha256":"wrong"}})


def test_v1_cache_gt_is_isolated_and_geometry_is_not_claimed_valid(tmp_path):
    rawdir=tmp_path/"raw";rawdir.mkdir()
    raw,_,_=fixture_cache(rawdir)
    root=tmp_path/"support";scene=root/"scene_08";scene.mkdir(parents=True)
    path=scene/"shard_0000.npz"
    rows=joined_rows(raw)
    np.savez(path,timestamps=[1],frame_offsets=[0,2],camera_ids=[1,2],detection_ids=[0,0],
        bbox=np.array([r["bbox"] for r in rows]),scores=np.array([r["score"] for r in rows]),
        center=np.ones((2,3)),covariance=np.array([np.eye(3)]*2),
        box_support=np.ones((2,3,3)),foreground_support=np.ones((2,3,3)),
        box_support_count=[3,3],foreground_support_count=[3,3],quality=[1.,1.],
        foreground_fraction=[1.,1.],ambiguity=[1,1],identity=[99,99])
    path.with_suffix(".json").write_text(json.dumps({"records":[{"image_ids":{"1":1,"2":2},"gt_id":99}]}))
    (scene/"manifest.json").write_text(json.dumps({"schema":V1,"status":"COMPLETE","timestamps":1,
        "detector_cache":raw.manifest(),"shards":[{"file":path.name,"sha256":sha256_file(path)}]}))
    with pytest.raises(FeatureUnavailable):
        WorldObjectCache(root,8)
    with pytest.raises(ValueError,match="REQUIRES_REBUILD"):
        WorldObjectCache(root,8,raw_cache=raw,require_geometry=True)
    frame=next(WorldObjectCache(root,8,raw_cache=raw).frames())
    assert "identity" not in frame["rows"][0] and "gt_id" not in frame["record"]
    observations=replay_observations(frame)
    assert not observations[0].geometry_valid and observations[0].feature_valid


def test_inference_adapter_refuses_target_fields():
    with pytest.raises(ValueError,match="GT_IDENTITY_FORBIDDEN"):
        replay_observations({"rows":[{"identity":4}],"timestamp":0})

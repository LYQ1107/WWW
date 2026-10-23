import hashlib
import json
import numpy as np
import pytest
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.person_detector import sha256_file


def fixture_cache(tmp_path):
    lock={'schema':'gmt_pre_gta_rgb_person_roi_v1','split':'train'}
    (tmp_path/'DETECTOR_LOCK.json').write_text(json.dumps(lock))
    scene=tmp_path/'scene_08';scene.mkdir()
    path=scene/'shard_0000.npz'
    np.savez(path,frame_offsets=np.array([0,2]),timestamps=np.array([1]),
        camera_ids=np.array([1,2]),detection_ids=np.array([0,0]),
        bbox=np.array([[1,1,3,4],[2,2,3,4]],np.float32),scores=np.array([.8,.9],np.float32),
        roi_features=np.arange(16,dtype=np.float32).reshape(2,8))
    im=np.zeros((8,8,3),np.uint8);digest=hashlib.sha256(im.tobytes()).hexdigest()
    record={'image_ids':{'1':1,'2':2},'rgb_sha256':{'1':digest,'2':digest}}
    path.with_suffix('.json').write_text(json.dumps({'records':[record]}))
    shard={'file':path.name,'sha256':sha256_file(path),'timestamps':[1]}
    (scene/'manifest.json').write_text(json.dumps({'status':'COMPLETE','signature':lock,'shards':[shard]}))
    return ImmutableGMTDetectionCache(tmp_path),im,path


def test_scene_timestamp_image_id_and_rgb_are_all_verified(tmp_path):
    cache,image,_=fixture_cache(tmp_path)
    with pytest.raises(RuntimeError,match='context'):cache.infer({1:image,2:image})
    cache.set_context(8,1,{1:1,2:2})
    out=cache.infer({1:image,2:image})
    assert len(out[1])==len(out[2])==1
    assert out[1][0].mask is None and out[1][0].source=='gmt_pre_gta_detector'
    assert np.array_equal(out[2][0].detector_roi_feature,np.arange(8,16))
    with pytest.raises(ValueError,match='Image IDs'):cache.get(8,1,image_ids={1:99,2:2})
    with pytest.raises(ValueError,match='RGB'):cache.get(8,1,{1:image+1,2:image})
    with pytest.raises(KeyError):cache.get(8,2)


def test_shard_checksum_failure_cannot_silently_replay(tmp_path):
    cache,_,path=fixture_cache(tmp_path)
    with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(RuntimeError,match='checksum'):cache.get(8,1)

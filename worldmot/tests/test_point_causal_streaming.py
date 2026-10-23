import pytest
from point_fakes import PointOracle,scene_detections,images
from worldmot.tracking.point_identity_memory import PointIdentityMemory


def run_prefix(count):
    oracle=PointOracle();memory=PointIdentityMemory(oracle)
    rows=[memory.step(images(),scene_detections(t),t,8) for t in range(1,count+1)]
    return [r['tracks_2d_by_camera'] for r in rows],memory,oracle


def test_prefix_outputs_are_unchanged_when_later_images_are_present():
    short,_,_=run_prefix(2);long,_,oracle=run_prefix(4)
    assert short==long[:2]
    assert all(tuple(keys[0])==reference and all(k[1]<=now for k in keys) for keys,reference,now in oracle.calls)


def test_repeated_timestamp_is_rejected_instead_of_polluting_ids():
    _,memory,_=run_prefix(1)
    with pytest.raises(ValueError,match='increasing'):
        memory.step(images(),scene_detections(),1,8)

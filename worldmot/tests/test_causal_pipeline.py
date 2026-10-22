import numpy as np

from worldmot.geometry.camera_model import CameraModel
from worldmot.models.geometry_interface import GeometryBackend, GeometryFrame
from worldmot.runtime.online_worldmot import OnlineWorldMOT


class FakeGeometry(GeometryBackend):
    def reconstruct(self, images, timestamp):
        pm = np.zeros((8, 8, 3), dtype=np.float64); pm[..., 2] = 5; pm[2:6, 2:4, :2] = [0, 0]
        conf = np.ones((8, 8)) * 2
        pan = np.zeros((8, 8), dtype=np.int32); pan[2:6, 2:4] = 1
        pred = {v: {"pan": pan, "segments_info": [{"id": 1, "query_id": 0, "category_id": 0}]} for v in images}
        cams = {v: CameraModel(np.array([[100., 0, 4], [0, 100., 4], [0, 0, 1]]), np.eye(4), (8, 8), (8, 8)) for v in images}
        return GeometryFrame(timestamp, images, {v: pm for v in images}, {v: conf for v in images}, pred,
                             np.array([[1., 0.]], dtype=np.float32), cams)


def test_online_step_is_causal_and_keeps_memory():
    runtime = OnlineWorldMOT(FakeGeometry(), min_points=3)
    runtime.reset("s")
    imgs = {1: np.zeros((8, 8, 3), np.uint8), 2: np.zeros((8, 8, 3), np.uint8)}
    first = runtime.step(imgs, 1)
    second = runtime.step(imgs, 2)
    assert first["causal"] and second["causal"]
    assert first["tracks_3d"][0]["global_id"] == second["tracks_3d"][0]["global_id"]
    assert all(x["projected_from_world"] for x in second["tracks_2d_by_camera"][1])

import numpy as np

from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer
from worldmot.models.world_observation_encoder import WorldObservationToken
from worldmot.tracking.persistent_query_runtime import PersistentQueryRuntime


def _token(camera_id: int, detection_id: int) -> WorldObservationToken:
    return WorldObservationToken(
        camera_id=camera_id, timestamp=1, detection_id=detection_id,
        image_feature=np.zeros(256, np.float32), omega_feature=np.zeros(64, np.float32),
        xyz_world=np.zeros(3, np.float32), covariance=np.eye(3, dtype=np.float32),
        bbox=np.array([1, 1, 2, 3], np.float32), mask_feature=np.zeros(64, np.float32),
        detection_score=0.9, geometry_reliability=1.0,
        geometry_feature=np.zeros(18, np.float32), valid=True, source="test",
    )


def test_runtime_is_causal_and_marks_untrained_projection_as_diagnostic():
    model = PersistentWorldQueryTransformer(hidden_dim=32, max_world_queries=4, heads=4, decoder_layers=1)
    runtime = PersistentQueryRuntime(model, device="cpu", allow_untrained=True)
    first = runtime.step([_token(1, 0), _token(2, 0)], 1)
    second = runtime.step([_token(1, 0), _token(2, 0)], 2)
    assert first.status == "SMOKE_UNTRAINED"
    assert second.timestamp == 2
    assert second.diagnostics["coordinates_calibrated"] is False
    assert len(set(second.global_ids_by_observation.values())) <= 2


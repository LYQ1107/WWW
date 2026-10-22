import json
from pathlib import Path

import pytest


def test_capacity_test_waits_for_original_test_and_final_training(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import run_capacity_test as runner
    original = tmp_path / 'original'
    capacity = tmp_path / 'capacity'
    original.mkdir()
    capacity.mkdir()
    monkeypatch.setattr(runner, 'ORIGINAL', original)
    (original / 'experiment_status.json').write_text(json.dumps({'stage': 'INDEPENDENT_TEST_RUNNING'}))
    (capacity / 'experiment_status.json').write_text(json.dumps({'stage': 'CAPACITY_ALL24_FITS_COMPLETE_AWAITING_INDEPENDENT_TEST'}))
    assert runner.ready(capacity) is False
    (original / 'experiment_status.json').write_text(json.dumps({'stage': 'RESULTS_READY_FOR_REVIEW_AND_GITHUB_PUSH'}))
    assert runner.ready(capacity) is True
    (capacity / 'experiment_status.json').write_text(json.dumps({'stage': 'CAPACITY_TRAIN_ONLY_SELECTING'}))
    assert runner.ready(capacity) is False


def test_capacity_collector_rejects_missing_video_streams(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from collect_capacity_test_results import collect
    lock = {'checkpoints': [{'profile': 'large', 'seed': s, 'sha256': 'test-only'}
                           for s in (20261022, 20261023, 20261024)] +
                           [{'profile': 'small', 'seed': 20261022, 'sha256': 'test-only'}]}
    (tmp_path / 'final_test_lock.json').write_text(json.dumps(lock))
    with pytest.raises(RuntimeError, match='rejects partial'):
        collect(tmp_path)

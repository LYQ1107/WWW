import csv
import json
import pytest
from sample_worldobject_v3_resources import FIELDS,measurements_ready,resource_log_mode


def test_atomic_state_replacement_or_partial_json_does_not_end_observation(tmp_path):
    status=tmp_path/"status.json"
    assert not measurements_ready(status)
    status.write_text('{"status":')
    assert not measurements_ready(status)
    status.write_text(json.dumps({"status":"RUNNING_REGISTERED_DEPENDENCIES"}))
    assert not measurements_ready(status)
    status.write_text(json.dumps({"status":"MEASUREMENTS_READY_FOR_SCIENTIFIC_AUDIT"}))
    assert measurements_ready(status)


def test_resource_resume_preserves_all_existing_rows_and_does_not_duplicate_header(tmp_path):
    path=tmp_path/"samples.csv"
    mode,header=resource_log_mode(path,False)
    with path.open(mode) as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        if header:writer.writeheader()
        writer.writerow(dict(time_unix=1,gpu_uuid="device",device_used_bytes=128))
    before=path.read_bytes();mode,header=resource_log_mode(path,True)
    with path.open(mode) as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        if header:writer.writeheader()
        writer.writerow(dict(time_unix=2,gpu_uuid="device",device_used_bytes=256))
    assert path.read_bytes().startswith(before)
    with path.open() as stream:rows=list(csv.DictReader(stream))
    assert [row["time_unix"] for row in rows]==["1","2"]
    with pytest.raises(RuntimeError,match="PRESERVE_EXISTING"):
        resource_log_mode(path,False)


def test_corrupt_or_partial_resource_logs_are_not_silently_overwritten(tmp_path):
    path=tmp_path/"samples.csv";path.write_text("wrong,header\n")
    with pytest.raises(RuntimeError,match="HEADER_MISMATCH"):resource_log_mode(path,True)
    path.write_text(",".join(FIELDS)+"\n1,partial")
    with pytest.raises(RuntimeError,match="INCOMPLETE_RESOURCE_LOG"):resource_log_mode(path,True)

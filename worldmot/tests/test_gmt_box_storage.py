import numpy as np
from worldmot.models.gmt_detector_adapter import bbox_xyxy_to_float32_xywh


def test_native_float32_xyxy_subtraction_matches_immutable_xywh_storage_exactly():
    # Real Scene 4 frame 598: subtraction in Python float preserves one more
    # bit than storage FP32; exact JSON equality must use the declared schema.
    box=np.array([77.76180267333984,554.2482299804688,
                  305.5873718261719,1008.8659057617188],np.float32)
    result=bbox_xyxy_to_float32_xywh(box)
    assert result==(77.76180267333984,554.2482299804688,227.8255615234375,454.61767578125)
    assert tuple(np.asarray(result,np.float32).tolist())==result
    assert result[2]!=float(box[2])-float(box[0])

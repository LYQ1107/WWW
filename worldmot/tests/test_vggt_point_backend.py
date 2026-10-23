import numpy as np
import pytest
from worldmot.models.vggt_point_backend import VGGTPointBackend


def test_first_image_is_actual_reference_and_no_future_is_accepted():
    images=[np.zeros((120,200,3),np.uint8)]*3
    points=np.array([[50,60]],np.float32)
    keys=[(1,4),(1,5),(2,5)]
    assert VGGTPointBackend.validate_request(images,points,keys,(1,4),5).shape==(1,2)
    with pytest.raises(ValueError,match='actual source'):
        VGGTPointBackend.validate_request(images,points,keys,(2,4),5)
    with pytest.raises(ValueError,match='Future'):
        VGGTPointBackend.validate_request(images,points,keys,(1,4),4)
    with pytest.raises(ValueError,match='source image'):
        VGGTPointBackend.validate_request(images,np.array([[201,60]]),keys,(1,4),5)


def test_empty_points_are_legal_but_nonfinite_points_are_rejected():
    image=[np.zeros((120,200,3),np.uint8)]
    assert VGGTPointBackend.validate_request(image,[],[(2,0)],(2,0),0).shape==(0,2)
    with pytest.raises(ValueError,match='Nonfinite'):
        VGGTPointBackend.validate_request(image,[[np.nan,0]],[(2,0)],(2,0),0)

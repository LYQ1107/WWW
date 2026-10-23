import numpy as np
import pytest
from worldmot.geometry.vggt_point_coordinates import VGGTPointCoordinates


@pytest.mark.parametrize('shape',[(1080,1920,3),(777,1023,3),(960,640,3),(518,518,3),(1440,2560,3)])
def test_roundtrip_and_padding(shape):
    t=VGGTPointCoordinates.from_shape(shape)
    q=np.array([[0,0],[shape[1]-1,shape[0]-1],[31.5,45.25]])
    assert np.max(np.abs(t.to_original(t.to_network(q))-q))<1e-9
    assert t.valid_network(t.to_network(q)).all()
    if t.pad_top:
        assert not t.valid_network(np.array([[t.pad_left,0]]))[0]
    if t.pad_left:
        assert not t.valid_network(np.array([[0,t.pad_top]]))[0]


def test_preprocessing_keeps_complete_view_and_rejects_wrong_rgb_shape():
    a=VGGTPointCoordinates.from_shape((90,161,3))
    image=np.full((90,161,3),[255,0,0],np.uint8)
    output=a.preprocess(image)
    assert output.shape==(3,518,518)
    assert np.array_equal(output[:,0,0],np.ones(3))
    assert np.array_equal(output[:,259,259],np.array([1,0,0]))
    with pytest.raises(ValueError):a.preprocess(image[:80])

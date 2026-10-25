import numpy as np
import pytest
from worldmot.models.world_object_support import extract_world_object_support,map_bbox_to_model
from worldmot.geometry.camera_model import CameraModel
from worldmot.geometry.world_alignment import Sim3
from worldmot.geometry.world_frame_contract import WorldFrameTransform
from worldmot.models.omega_geometry_backend import PixelMapping,_unproject_depth


def test_rotated_camera_clusters_camera_depth_not_world_z():
    # Camera forward points along world +x. The near surface has LARGE world z.
    pm=np.zeros((8,8,3));pm[:,:4]=[2,0,20];pm[:,4:]=[9,0,1]
    camera_from_world=np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1.]])
    support=extract_world_object_support(pm,np.full((8,8),2.),[0,0,8,8],
                                        camera_from_world=camera_from_world,interior_margin=0)
    np.testing.assert_allclose(support.foreground_points[:,0],2)
    assert support.depth_source=="CAMERA_FROM_OMEGA_LOCAL_WORLD"


def test_no_pose_is_explicitly_unavailable():
    support=extract_world_object_support(np.ones((4,4,3)),np.ones((4,4))*2,[0,0,4,4])
    assert not support.reliable and support.ambiguity=="CAMERA_DEPTH_UNAVAILABLE"


@pytest.mark.parametrize("value",[0.,-1.,np.nan,np.inf])
def test_invalid_or_behind_camera_depth_is_filtered(value):
    pm=np.ones((8,8,3));pm[:,:,2]=value
    s=extract_world_object_support(pm,np.ones((8,8))*2,[0,0,8,8],camera_from_world=np.eye(4))
    assert not s.reliable and len(s.foreground_points)==0


def test_camera_pose_translation_scale_and_sim3_covariance():
    pose=np.eye(4);pose[:3,3]=[1,2,3]
    camera=CameraModel(np.eye(3),pose,(8,8),(8,8),np.eye(3))
    np.testing.assert_allclose(camera.forward_depth([[1,2,8]]),[5])
    theta=.7;r=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    transform=WorldFrameTransform(Sim3(2,r,np.ones(3)),4,True)
    cov=np.diag([1.,2.,3.])
    np.testing.assert_allclose(transform.covariance(cov),4*r@cov@r.T)
    points=np.array([[2.,3.,4.]])
    inverse=Sim3(.5,r.T,-r.T@np.ones(3)/2)
    np.testing.assert_allclose(inverse.apply(transform.apply(points)),points)
    with pytest.raises(ValueError,match="WORLD_ALIGNMENT_FAILED"):
        WorldFrameTransform(Sim3(),4,False).apply(points)


def test_bbox_crop_resize_pad_and_outside_projection():
    mapping=PixelMapping(100,80,10,5,80,60,40,30,4,3,48,36)
    x,y,w,h=map_bbox_to_model([10,5,80,60],mapping.matrix)
    assert np.allclose([x,y,w,h],[3.75,2.75,40,30])
    outside=map_bbox_to_model([200,200,10,10],mapping.matrix,(36,48))
    assert np.array_equal(outside,[48,36,0,0])
    camera=CameraModel(np.eye(3),np.eye(4),(8,8),(8,8),np.eye(3))
    _,valid=camera.project([[100,100,1],[1,1,-1]],clip=True)
    assert not valid.any()


def test_depth_unprojection_extrinsic_direction_and_pixel_alignment():
    ext=np.concatenate([np.eye(3),np.array([[0],[0],[-3.]])],axis=1)[None]
    pm=_unproject_depth(np.ones((1,2,2))*2,ext,np.eye(3)[None])
    np.testing.assert_allclose(pm[0,0,0],[0,0,5])
    with pytest.raises(ValueError,match="PIXEL_ALIGNMENT"):
        extract_world_object_support(pm[0],np.ones((2,2))*2,[0,0,2,2],depth_map=np.ones((3,3)))

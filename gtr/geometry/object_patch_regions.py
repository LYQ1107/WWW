"""Actual patch centers and bbox-edge coordinates; no interpolation or copied points."""
import numpy as np


def patch_lattice(mapping,patch_size=16):
    if mapping.network_width%patch_size or mapping.network_height%patch_size:
        raise ValueError('network extent must be a whole patch grid')
    h,w=mapping.network_height//patch_size,mapping.network_width//patch_size
    y,x=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    left=x.ravel()*patch_size;top=y.ravel()*patch_size
    centers=np.c_[left+patch_size/2-.5,top+patch_size/2-.5]
    original,center_valid=mapping.inverse(centers)
    valid=center_valid & (left>=mapping.pad_left)&(top>=mapping.pad_top)
    valid &= (left+patch_size<=mapping.pad_left+mapping.resize_width)&(top+patch_size<=mapping.pad_top+mapping.resize_height)
    return dict(grid_shape=(h,w),patch_ids=np.arange(h*w),network_centers=centers,
                original_centers=original,valid=valid,padding_or_partial_padding=~valid)


def object_regions(mapping,bbox_xyxy,patch_size=16,K_max=16):
    box=np.asarray(bbox_xyxy,dtype=float)
    if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):raise ValueError('invalid native bbox')
    if not isinstance(K_max,int) or K_max<1:raise ValueError('positive unique-token cap required')
    lattice=patch_lattice(mapping,patch_size)
    center=(box[:2]+box[2:])/2;half=(box[2:]-box[:2])/2
    extended=np.r_[center-1.5*half,center+1.5*half]
    def map_edges(b):
        points=np.array([[b[0],b[1],1],[b[2],b[3],1]])@mapping.A_edges.T
        return points[:,:2].reshape(4)
    network_box=map_edges(box);network_context=map_edges(extended)
    ce=lattice['network_centers']+.5
    inside=lambda b:(ce[:,0]>=b[0])&(ce[:,0]<b[2])&(ce[:,1]>=b[1])&(ce[:,1]<b[3])
    target=inside(network_box)&lattice['valid'];context=inside(network_context)&lattice['valid']
    masks={'M0':target,'M1':context,'M2':context&~target}
    regions={}
    for name,mask in masks.items():
        all_ids=np.flatnonzero(mask)
        ids=all_ids[np.linspace(0,len(all_ids)-1,min(K_max,len(all_ids)),dtype=int)] if len(all_ids) else all_ids
        assert len(np.unique(ids))==len(ids)
        regions[name]=dict(patch_ids=ids,available_count=len(all_ids),selected_count=len(ids),
            network_centers=lattice['network_centers'][ids],original_centers=lattice['original_centers'][ids],
            is_object_token=target[ids],object_token_fraction=float(target[ids].mean()) if len(ids) else None,
            region_type='OBJECT_REGION' if name=='M0' else 'BACKGROUND_CONTEXT',
            point_correspondence_available=len(ids)>=2)
    n=int(target.sum());bucket='0' if n==0 else '1' if n==1 else '2-3' if n<4 else '4-8' if n<9 else '9+'
    return dict(original_bbox=box,original_bbox_area=float(np.prod(box[2:]-box[:2])),network_bbox=network_box,
        network_bbox_size=network_box[2:]-network_box[:2],valid_object_patch_count=n,
        valid_context_patch_count=int(context.sum()),patch_bucket=bucket,regions=regions)

"""Validation-only swap of encoder inputs with physical camera restoration."""


def reverse_and_restore(extract,images,boxes,*args,**kwargs):
    view_dimension=0 if images.ndim==4 else 1 if images.ndim==5 else None
    if view_dimension is None or images.shape[view_dimension]!=2 or len(boxes)!=2:
        raise ValueError('view-order validation requires one two-camera pair')
    result,coverage,metadata=extract(images.flip(view_dimension),boxes[::-1],*args,**kwargs)
    result={key:values[::-1] for key,values in result.items()}
    coverage={1-view:value for view,value in coverage.items()}
    metadata=dict(metadata)
    if metadata.get('pose') is not None:metadata['pose']=metadata['pose'].flip(1)
    if metadata.get('patch_maps') is not None:
        metadata['patch_maps']={key:values[::-1] for key,values in metadata['patch_maps'].items()}
    return result,coverage,metadata

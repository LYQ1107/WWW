"""Validation-only correction of the original per-image camera scalar input."""


def install_single_image_camera_control(model):
    if not model.roi_heads.concat_emb:raise ValueError('camera control requires original concatenated spatial/camera head')
    current=[None];inference=model.inference
    def set_camera(module,inputs):
        if current[0] is None:raise RuntimeError('camera hook called outside the declared per-image inference')
        values=inputs[0].clone()
        if values.ndim!=2 or values.shape[1]!=11:raise ValueError('unexpected spatial/camera input shape')
        values[:,-1]=current[0]
        return (values,*inputs[1:])
    handle=model.roi_heads.s_t_head.register_forward_pre_hook(set_camera)
    def controlled(batch,view_num,time,view,**kwargs):
        if len(batch)!=1 or view_num<1 or not 0<=view<view_num:raise ValueError('declared single-image view index required')
        current[0]=view/float(view_num)
        try:return inference(batch,view_num,time,view,**kwargs)
        finally:current[0]=None
    model.inference=controlled
    return handle

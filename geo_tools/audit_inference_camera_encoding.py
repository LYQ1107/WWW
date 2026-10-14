"""Probe the actual official ROI method's single-image camera input semantics."""
import types
import torch
from common import *
from detectron2.structures import Instances,Boxes
from gtr.modeling.roi_heads.gtr_roi_heads import GTRROIHeads


def probe(view):
    capture=[]
    model=types.SimpleNamespace(asso_on=True,asso_thresh_test=-1.,training=False,asso_in_features=['p3'],feature_dim=4,cat_dim=2,concat_emb=True)
    model.jitter_bboxes_center=types.MethodType(GTRROIHeads.jitter_bboxes_center,model)
    model.asso_pooler=lambda features,boxes:torch.ones(sum(len(box) for box in boxes),4,1,1)
    model.asso_head=lambda values:values.flatten(1)
    def spatial(values):capture.append(values.detach().clone());return values.new_zeros((len(values),2))
    model.s_t_head=spatial
    inst=Instances((100,100));inst.proposal_boxes=Boxes(torch.tensor([[5.,5,15,25],[40,40,60,60]]));inst.objectness_logits=torch.ones(2)
    output=GTRROIHeads._forward_asso(model,{'p3':torch.ones(1,4,20,20)},[inst],2,[1,1,100],view)
    return dict(requested_view=view,view_num=2,batch_instances=1,observed_camera_scalar=capture[0][:,-1].tolist(),reid_width=output[0].reid_features.shape[1])


def main():
    seed_all(20261014);rows=[probe(view) for view in [0,1]]
    save_json(ROOT/'results/inference_camera_encoding_source_probe_v1.json',dict(status='INFERENCE_CAMERA_COLLAPSE_OBSERVED' if rows[0]['observed_camera_scalar']==rows[1]['observed_camera_scalar'] else 'DISTINCT_CAMERA_SCALARS',rows=rows,
        scope='actual GTRROIHeads._forward_asso implementation with stubbed perception/MLP capture on CPU; semantic camera input proof, not a GPU accuracy test',
        source_sha256=sha256(ROOT/'gtr/modeling/roi_heads/gtr_roi_heads.py'),
        explanation='official loop computes num_frame=int(len(instances)/view_num); with one image and two views this is0, so all assignments are empty slices; supplied view argument is not used for this field',
        limitations='locked production baseline is unchanged; full live impact and any explicit corrected-camera control require separate evaluation; existing paired results remain results of their declared official-source protocol',provenance=provenance()))


if __name__=='__main__':main()

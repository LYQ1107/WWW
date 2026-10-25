from collections import Counter
from worldmot.evaluation.synchronous_multiview_identity import match_boxes,events,update_counts,summarize,paired_counts


def gt(ident,box):return dict(instance_id=ident,bbox=box,category_id=1)
def track(camera,gid,box,det=0):return dict(camera_id=camera,global_id=gid,bbox_xywh=box,detection_id=det)


def test_same_protocol_different_detection_order_uses_gt_event_keys():
    truth={1:[gt(1,[0,0,10,10])],2:[gt(1,[0,0,10,10]),gt(2,[20,0,10,10])]}
    left,_=match_boxes([track(1,8,[0,0,10,10]),track(2,8,[0,0,10,10]),track(2,9,[20,0,10,10],1)],truth)
    right,_=match_boxes([track(2,6,[20,0,10,10],7),track(1,5,[0,0,10,10],9),track(2,5,[0,0,10,10],4)],truth)
    assert set(left)==set(right)
    c,_=paired_counts(left,right);assert c=={'both_correct':2}


def test_protocol_b_intersects_gt_keys_not_prediction_id_or_ordinal():
    truth={1:[gt(1,[0,0,10,10])],2:[gt(1,[0,0,10,10]),gt(2,[20,0,10,10])]}
    a,_=match_boxes([track(1,1,[0,0,10,10]),track(2,2,[0,0,10,10]),track(2,3,[20,0,10,10],1)],truth)
    b,_=match_boxes([track(1,9,[0,0,10,10]),track(2,9,[0,0,10,10])],truth)
    c,kind=paired_counts(a,b);assert c['right_corrects_left']==1
    assert kind['different_person']=={} and len(list(events(a)))==2 and len(list(events(b)))==1


def test_duplicate_gt_identity_masks_overlapping_predictions():
    truth={1:[gt(1,[0,0,10,10]),gt(1,[20,0,10,10])]}
    matched,audit=match_boxes([track(1,2,[0,0,10,10]),track(1,3,[20,0,10,10],1)],truth)
    assert not matched and audit['ambiguity_mask_indices']==[0,1]
    assert audit['counts']['ambiguous_or_invalid_GT']==2


def test_synchronous_metrics_do_not_hide_same_person_failure_in_accuracy():
    c=Counter()
    for _ in range(1000):update_counts(c,False,False)
    update_counts(c,True,False)
    m=summarize(c);assert m['Recall_crossview']==0 and m['F1_crossview']==0
    assert m['False_Merge_Rate']==0 and m['Precision_crossview'] is None


def test_one_to_one_match_does_not_assign_two_boxes_to_same_gt():
    truth={1:[gt(1,[0,0,10,10])]}
    matched,audit=match_boxes([track(1,1,[0,0,10,10]),track(1,2,[1,0,10,10],1)],truth)
    assert len(matched)==1 and matched[1,'1']['global_id']==1 and audit['counts']['unmatched_detections']==1


def test_matching_maximizes_legal_cardinality_before_iou_sum():
    truth={1:[gt(1,[0,0,10,10]),gt(2,[4,0,10,10])]}
    # Greedy A->GT1 would erase B's only valid edge. Fair matching gives both.
    matched,_=match_boxes([track(1,10,[1.5,0,10,10]),track(1,11,[-2,0,10,10],1)],truth)
    assert len(matched)==2 and matched[1,'1']['global_id']==11 and matched[1,'2']['global_id']==10


def test_estimated_camera_axis_angle_is_not_inferred_from_roi_cosine(monkeypatch):
    import importlib
    from pathlib import Path
    import numpy as np
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'analysis'))
    module=importlib.import_module('compare_gmt_v4_synchronous_identity')
    a=np.eye(4);b=np.eye(4);b[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
    assert abs(module.camera_axis_angle({'camera_models':{'1':{'world_from_camera':a.tolist()},'2':{'world_from_camera':b.tolist()}}})-90)<1e-8
    assert module.camera_axis_angle({}) is None

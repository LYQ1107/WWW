"""Post-hoc TRAIN window objective, explicitly distinct from official MOT metrics."""
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from worldmot.models.person_detector import PersonDetection
from worldmot.training.identity_targets import match_detections_to_gt


def score_rows(rows,gt_dataset,scene):
    counts=Counter();confusion=Counter();last={}
    for row in rows:
        detections={int(v):[PersonDetection(int(v),d['detection_id'],tuple(d['bbox_xywh']),d['score']) for d in values]
            for v,values in row['detections_by_camera'].items()}
        gt={int(v):gt_dataset.ground_truth(int(i)) for v,i in row['image_ids'].items()}
        targets=match_detections_to_gt(detections,gt,scene)
        counts['raw_GT_boxes']+=sum(map(len,gt.values()));counts['raw_candidates']+=sum(map(len,detections.values()))
        by_view=defaultdict(dict)
        for v,values in row['tracks_2d_by_camera'].items():
            view=int(v)
            for out in values:
                index=next(i for i,d in enumerate(detections[view]) if d.detection_id==out['detection_id'])
                target=targets[view,index].identity_key;gid=out['global_id'];counts['output_boxes']+=1
                if target is None:counts['unmatched_or_ambiguous_boxes']+=1;continue
                confusion[target,gid]+=1;counts['matched_boxes']+=1;by_view[view][target]=gid
                if (view,target) in last and last[view,target]!=gid:counts['diagnostic_temporal_ID_changes']+=1
                last[view,target]=gid
        if len(by_view)==2:
            a,b=sorted(by_view)
            for target in by_view[a].keys()&by_view[b].keys():
                counts['same_GT_both_views_output']+=1
                if by_view[a][target]==by_view[b][target]:counts['correct_cross_view_shared_ID']+=1
            for ta,ga in by_view[a].items():
                for tb,gb in by_view[b].items():
                    if ga==gb and ta!=tb:counts['different_GT_cross_view_wrong_merge']+=1
    gt_ids=sorted({k[0] for k in confusion});pred_ids=sorted({k[1] for k in confusion})
    matrix=np.zeros((len(gt_ids),len(pred_ids)),np.int64)
    gi={k:i for i,k in enumerate(gt_ids)};pi={k:i for i,k in enumerate(pred_ids)}
    for (target,gid),n in confusion.items():matrix[gi[target],pi[gid]]=n
    rr,cc=linear_sum_assignment(-matrix);counts['window_identity_true_positive']=int(matrix[rr,cc].sum())
    counts['window_identity_false_positive']=counts['output_boxes']-counts['window_identity_true_positive']
    counts['window_identity_false_negative']=counts['raw_GT_boxes']-counts['window_identity_true_positive']
    assert counts['output_boxes']==counts['raw_candidates'],'Shared candidate coverage changed'
    return dict(counts)


def rank_record(counts,settings):
    tp=counts.get('window_identity_true_positive',0)
    denominator=counts.get('output_boxes',0)+counts.get('raw_GT_boxes',0)
    f1=2*tp/denominator if denominator else 0.
    return (f1,-counts.get('different_GT_cross_view_wrong_merge',0),
        -counts.get('diagnostic_temporal_ID_changes',0),settings.get('minimum_confidence',settings.get('minimum_cosine',0)),
        settings.get('minimum_support',settings.get('minimum_iou',0)),-settings.get('points_per_person',0))


OBJECTIVE={'name':'TRAIN_window_identity_consistency_F1','formula':'2 * max bipartite GT/pred-ID matched box count / (all output boxes + all original GT boxes), summed over independent windows',
    'IoU':.5,'official_MOT_metric':False,'raw_GT_modified':False,
    'tie_break':'fewer cross-view wrong merges, fewer temporal ID changes, higher confidence/support thresholds, fewer points',
    'duplicate_GT':'Ambiguous duplicate identity labels are not assigned to candidates; all raw GT boxes stay in the denominator.'}

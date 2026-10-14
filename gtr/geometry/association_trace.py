"""Optional counterfactual diagnostics; never used to choose tracker decisions.

Tracing synchronizes tensors and its runtime is unsuitable for speed claims.
Before/after comparisons share the current live history, not separate runs.
"""
import torch
from scipy.optimize import linear_sum_assignment

def association_effect(before, after, instances, k, roi_heads, overlap_thresh, not_mult_thresh):
    refs = [x for i, x in enumerate(instances) if i != k]
    counts = [len(x) for x in refs]
    ids = torch.cat([x.track_ids for x in refs])
    unique = ids.unique()
    incidence = (ids[:, None] == unique[None, :]).float()
    scores = []
    for logits in [before, after]:
        # Exactly the original global GTA: per historical-frame unmatched
        # softmax, then sum the probabilities of each historical identity.
        probability = torch.cat(roi_heads._activate_asso(logits.split(counts, dim=1)), dim=1)
        scores.append(probability @ incidence)
    decisions = []
    for score in scores:
        chosen = ids.new_full((len(instances[k]),), -1)
        rows, cols = linear_sum_assignment(-score.detach().cpu().numpy())
        for row, col in zip(rows, cols):
            threshold = overlap_thresh if not_mult_thresh else overlap_thresh * incidence[:, col].sum()
            if score[row, col] > threshold:
                chosen[row] = unique[col]
        decisions.append(chosen)
    delta = (after - before).detach()
    result = dict(query_count=len(instances[k]), history_count=len(ids), track_count=len(unique),
                  changed_logit_pairs=int((delta != 0).sum()),
                  max_abs_logit_delta=float(delta.abs().max()) if delta.numel() else 0.,
                  max_abs_gta_delta=float((scores[1] - scores[0]).abs().max()) if scores[0].numel() else 0.,
                  counterfactual_assignment_changes=int((decisions[0] != decisions[1]).sum()))
    if len(unique):
        result['gta_top1_changes'] = int((scores[0].argmax(1) != scores[1].argmax(1)).sum())
    else:
        result['gta_top1_changes'] = 0
    return result

class AssociationTrace:
    def __init__(self, model):
        self.model = model
        self.original = model.geo_runtime.adapt
        self.records = []

    def adapt(self, outputs, instances, k):
        result = self.original(outputs, instances, k)
        record = association_effect(outputs[0], result[0], instances, k, self.model.roi_heads,
                                    self.model.overlap_thresh, self.model.not_mult_thresh)
        query = instances[k]
        if len(query) and query.has('geo_frame'):
            record.update(frame=int(query.geo_frame[0]), camera=int(query.geo_camera[0]),
                          valid_query_features=int(query.geo_valid.sum()))
            sources=query.geo_feature_frame if query.has('geo_feature_frame') else query.geo_frame
            valid=query.geo_valid.detach().cpu().tolist()
            ages=(query.geo_frame-sources).detach().cpu().tolist()
            record.update(query_feature_valid=valid,
                          feature_source_frames=sources.detach().cpu().tolist(),
                          feature_ages=[age if legal else None for age,legal in zip(ages,valid)],
                          query_boxes_original_xyxy=query.geo_xyxy.detach().cpu().tolist())
        self.records.append(record)
        return result

    def install(self):
        self.model.geo_runtime.adapt = self.adapt

    def close(self):
        self.model.geo_runtime.adapt = self.original

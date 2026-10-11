"""Strict CLEAR only where the actual prefix and future have unique GT IDs."""
from jev_phase15_causal_metrics import FrozenClearFuture
import collections

class AuditedClearFuture(FrozenClearFuture):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.duplicates=[]
  for view,data in self.templates.items():
   for frame,ids in enumerate(data['gt_ids']):
    duplicate=[int(i) for i,n in collections.Counter(ids.tolist()).items() if n>1]
    if duplicate:self.duplicates.append(dict(view=view,frame_zero_based=frame,IDs=duplicate))
 def horizon(self,ids,boundary,horizon,total_frames):
  end=min(total_frames,boundary[0]+horizon)
  invalid=[d for d in self.duplicates if d['frame_zero_based']<end]
  if not invalid:return super().horizon(ids,boundary,horizon,total_frames)
  return dict(future_CLEAR_IDSW=None,cumulative_CLEAR_IDSW=None,unchanged_prefix_CLEAR_IDSW=None,
   per_camera_future_CLEAR_IDSW=None,scene_frame_horizon_requested=horizon,
   scene_frames_observed=end-boundary[0],right_censored_at_video_end=end<boundary[0]+horizon,
   matching_history_retained_from_frame_zero=True,
   evaluator=dict(self.evaluator,strict_clear_causal_gate_eligible=False,
    status='NOT_EVALUABLE_DUPLICATE_GT_IN_PREFIX_OR_FUTURE',actual_duplicate_timesteps=invalid,
    raw_GT_unchanged=True,permissive_evaluation=False))

"""Explicit input contract for a frozen tracker with sequential camera updates.

Keep the original tracker unchanged. Canonicalize by physical view_id before
its complete step, and return outputs in the caller's original record order.
This is an interface guarantee, not internal query-order invariance.
"""
import numbers


class PhysicalCameraGMT:
    def __init__(self,model,tracker_factory=None):
        if tracker_factory is None:
            from .online_runner import CausalGMT
            tracker_factory=CausalGMT
        self.tracker=tracker_factory(model);self.camera_ids=None

    def __getattr__(self,name):
        return getattr(self.tracker,name)

    def step(self,records):
        if not records:raise ValueError('empty camera group')
        views=[record.get('view_id') for record in records]
        if any(not isinstance(view,numbers.Integral) or isinstance(view,bool) for view in views):
            raise ValueError('explicit integer physical view_id required')
        if len(set(views))!=len(views):raise ValueError('duplicate physical camera')
        membership=tuple(sorted(views))
        if self.camera_ids is not None and membership!=self.camera_ids:
            raise ValueError('physical camera membership changed')
        order=sorted(range(len(records)),key=lambda index:views[index])
        output=self.tracker.step([records[index] for index in order])
        if len(output)!=len(order):raise ValueError('tracker camera output count changed')
        restored=[None]*len(order)
        for slot,index in enumerate(order):restored[index]=output[slot]
        self.camera_ids=membership
        return restored

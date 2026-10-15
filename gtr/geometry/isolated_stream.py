"""Independent GMT banks and RNG streams while sharing frozen model weights."""
import random
import numpy as np
import torch
from .online_runner import CausalGMT
from .runtime import GeometryRuntime
from ..modeling.meta_arch.gtr_rcnn import poss_ids, old_ids, old_reids


def capture():
    return (poss_ids.poss_ids, old_ids.old_ids, old_reids.old_reids,
            random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore(state):
    poss_ids.poss_ids, old_ids.old_ids, old_reids.old_reids = state[:3]
    random.setstate(state[3])
    np.random.set_state(state[4])
    torch.set_rng_state(state[5])
    if state[6] is not None:
        torch.cuda.set_rng_state(state[6])


class IsolatedCausalStream:
    def __init__(self, model, runtime_factory=None):
        if model.geo_runtime.active:
            raise ValueError('shared tracking model must use GEO OFF; retrieval extraction is separate')
        self.model = model
        outer, runtime = capture(), model.geo_runtime
        try:
            self.runtime = (runtime_factory or GeometryRuntime)(runtime.cfg)
            model.geo_runtime = self.runtime
            self.tracker = CausalGMT(model)
            self.state = capture()
        finally:
            restore(outer)
            model.geo_runtime = runtime

    def step(self, records):
        outer, runtime = capture(), self.model.geo_runtime
        try:
            restore(self.state)
            self.model.geo_runtime = self.runtime
            output = self.tracker.step(records)
            self.state = capture()
            return output
        finally:
            restore(outer)
            self.model.geo_runtime = runtime

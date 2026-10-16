"""Bounded branch snapshots, preserving tensor/container aliasing via one deepcopy.

Restoration replaces tracker-owned state roots; callers must not retain mutable
references into an abandoned branch. Model parameters are frozen, not copied.
"""
import copy
import hashlib
import random
import numpy as np
import torch


def fingerprint(value):
    """Content digest; alias topology is tested separately, not implied by this hash."""
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((str(x.dtype), tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif hasattr(x, 'get_fields'):
            visit(x.image_size)
            visit(x.get_fields())
        elif hasattr(x, 'tensor'):
            visit(x.tensor)
        elif isinstance(x, dict):
            for k in sorted(x, key=repr):
                visit(k)
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for item in x:
                visit(item)
        elif isinstance(x, set):
            visit(sorted(x))
        elif isinstance(x, np.ndarray):
            h.update(str((str(x.dtype), x.shape)).encode())
            h.update(x.tobytes())
        else:
            h.update(repr(x).encode())
        h.update(b'\x00')

    visit(value)
    return h.hexdigest()


def native_state(tracker, module):
    return dict(tracker={k:v for k,v in vars(tracker).items() if k != 'model'},
                poss_ids=module.poss_ids.poss_ids, old_ids=module.old_ids.old_ids,
                old_reids=module.old_reids.old_reids,
                cir=vars(tracker.model.cir_controller))


class StateTransaction:
    def __init__(self, tracker, module):
        self.tracker, self.module = tracker, module
        # One memo across every root preserves native shared Tensor storage.
        self.snapshot = copy.deepcopy(native_state(tracker, module))
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.cpu_rng = torch.get_rng_state().clone()
        # Never initialize CUDA as a side effect of a CPU audit.
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        self.before_digest = fingerprint(self.snapshot)

    def restore(self):
        state = copy.deepcopy(self.snapshot)
        for key in list(vars(self.tracker)):
            if key != 'model':
                delattr(self.tracker, key)
        vars(self.tracker).update(state['tracker'])
        self.module.poss_ids.poss_ids = state['poss_ids']
        self.module.old_ids.old_ids = state['old_ids']
        self.module.old_reids.old_reids = state['old_reids']
        vars(self.tracker.model.cir_controller).clear()
        vars(self.tracker.model.cir_controller).update(state['cir'])
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        if fingerprint(native_state(self.tracker, self.module)) != self.before_digest:
            raise RuntimeError('CIR native rollback content mismatch')

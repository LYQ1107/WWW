"""Experimental single-use copy-on-write snapshot; not enabled in native runs.

Container topology is cloned eagerly. Original tensor storages are saved only
before a mutating ATen operation. The full StateTransaction remains the oracle.
"""
import copy
from collections import deque
import random
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from .state_transaction import native_state


def _tensors(root):
    seen = set()
    def visit(x):
        if id(x) in seen:
            return
        seen.add(id(x))
        if isinstance(x, torch.Tensor):
            yield x
        elif isinstance(x, dict):
            for k, v in x.items():
                yield from visit(k)
                yield from visit(v)
        elif isinstance(x, (tuple, list, set, deque)):
            for v in x:
                yield from visit(v)
        elif isinstance(x, torch.nn.Module) or (hasattr(x, '__dict__') and not callable(x)):
            yield from visit(vars(x))
    yield from visit(root)


def _key(t):
    return (str(t.device), t.untyped_storage()._cdata)


def _bytes(storage, device):
    return torch.empty(0, dtype=torch.uint8, device=device).set_(
        storage, 0, (storage.nbytes(),), (1,))


class NativeCommitTransaction(TorchDispatchMode):
    def __init__(self, tracker, module):
        super().__init__()
        self.tracker, self.module = tracker, module
        self.used = False
        self.saved = {}
        self.protected = {}
        original = native_state(tracker, module)
        memo = {}
        for t in _tensors(original):
            if t.layout != torch.strided:
                raise ValueError('COW snapshot supports strided tensors only')
            alias = t.detach().as_strided(t.size(), t.stride(), t.storage_offset())
            if isinstance(t, torch.nn.Parameter):
                alias = torch.nn.Parameter(alias, requires_grad=False)
            memo[id(t)] = alias
            self.protected[_key(t)] = (t.untyped_storage(), t.device)
        self.snapshot = copy.deepcopy(original, memo)
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func._schema.is_mutable:
            writes = []
            for i, arg in enumerate(func._schema.arguments):
                if arg.alias_info is not None and arg.alias_info.is_write:
                    value = args[i] if i < len(args) else kwargs.get(arg.name)
                    writes.extend(_tensors(value))
            # Check every destination before permitting any mutation.
            for t in writes:
                if _key(t) in self.protected and func._schema.name in (
                        'aten::resize_', 'aten::resize_as_', 'aten::set_'):
                    raise RuntimeError('COW protected storage resize/rebind rejected')
            with _disable_current_modes():
                for t in writes:
                    key = _key(t)
                    if key in self.protected and key not in self.saved:
                        storage, device = self.protected[key]
                        self.saved[key] = _bytes(storage, device).clone()
        return func(*args, **kwargs)

    def restore(self):
        if self.used:
            raise RuntimeError('COW restore is single-use')
        self.used = True
        with _disable_current_modes(), torch.inference_mode():
            for key, saved in self.saved.items():
                storage, device = self.protected[key]
                _bytes(storage, device).copy_(saved)
        state = self.snapshot
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

"""Compose admission with the unchanged native XV association/commit executor."""
from gtr.modeling.jev_phase15.native_commit_adapter import attach as attach_xv
from .safe_matching_memory import SafeMatchingMemory


def attach(model, rule='bounded_quarantine', prefix=None):
    executor = attach_xv(model)
    executor.memory_factory = lambda: SafeMatchingMemory(rule, prefix)
    executor.reset()
    original_commit = executor.commit

    def commit(*args, **kwargs):
        galleries = args[5] if len(args) > 5 else kwargs['galleries']
        executor.memory.raw_galleries = galleries
        executor.memory.last_admission = []
        return original_commit(*args, **kwargs)

    executor.commit = commit
    return executor

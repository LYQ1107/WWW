"""Use the unchanged XV Gallery/Bank commit, replacing only MATCH plan selection."""
import contextlib
from gtr.modeling.jev_phase15.native_commit_adapter import attach
import gtr.modeling.jev_stage2.native as original_native


def attach_jevtrack(model):
    executor=attach(model)
    original_scores=executor.scores;original_commit=executor.commit
    pending=[None]
    def scores(batch,refs,context,task):
        value=original_scores(batch,refs,context,task)
        if task==0:
            choice=getattr(executor.policy,'runtime_choice',None)
            pending[0]=list(choice) if choice is not None else None
        else:pending[0]=None
        return value
    executor.scores=scores
    def commit(*args,**kwargs):
        old=original_native.lawful_choice
        def choose(logits,legal):
            choice=pending[0];pending[0]=None
            if choice is None:return old(logits,legal)
            assert len(choice)==len(legal)
            assert len([c for c in choice if c>=0])==len({c for c in choice if c>=0})
            assert all(c<0 or bool(legal[r,c]) for r,c in enumerate(choice))
            return choice
        # Each actor is a separate, single-threaded process. Restore the original
        # module function even on exceptions. The plan generator imports its
        # canonical solver from assignment.py, so it never calls this shim.
        original_native.lawful_choice=choose
        try:return original_commit(*args,**kwargs)
        finally:original_native.lawful_choice=old;pending[0]=None
    executor.commit=commit
    return executor

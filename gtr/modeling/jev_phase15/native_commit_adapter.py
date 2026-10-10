"""Reuse the unchanged native executor and its actual Gallery/Bank lifecycle."""
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory
from .commitment_state import CommitmentState


class CommitmentMemory(CachedIdentityMemory):
    def __init__(self):
        super().__init__()
        self.commitment = CommitmentState()

    def update(self, ref, feature, box, image_size, frame, view):
        self.commitment.update(ref, feature, box, image_size, frame, view)
        return super().update(ref, feature, box, image_size, frame, view)

    def build(self, current, refs, galleries, frame, view, active, **kwargs):
        self.commitment.last_payload = int(frame), int(view)
        x = super().build(current, refs, galleries, frame, view, active, **kwargs)
        x['commitment_features'] = self.commitment.features(current, refs, frame, view)
        return x

    def state_dict(self):
        return dict(version=1, identity_memory=super().state_dict(), commitment_state=self.commitment.state_dict())

    def load_state_dict(self, state):
        assert state['version'] == 1
        super().load_state_dict(state['identity_memory'])
        self.commitment.load_state_dict(state['commitment_state'])


class FrozenOriginalPolicy:
    """Observe new state while retaining the exact original causal score function."""
    def __init__(self, policy): self.policy = policy
    def __call__(self, x): return self.policy({k: v for k, v in x.items() if k != 'commitment_features'})


def attach(model, policy=None):
    executor = model.jev_stage2_executor
    assert executor.mode == 'JEV_DIRECT'
    executor.memory_factory = CommitmentMemory
    executor.reset()
    if policy is not None: executor.policy = policy
    original_scores = executor.scores
    original_commit = executor.commit
    pending = {}

    def scores(batch, refs, context, task):
        result = original_scores(batch, refs, context, task)
        details = getattr(executor.policy, 'runtime_details', None)
        if task == 0 and details is not None and 'purity_logits' in details:
            pending[context['frame'], context['view']] = (refs.copy(), details['purity_logits'][0].sigmoid().cpu())
        return result

    def commit(*args, **kwargs):
        result = original_commit(*args, **kwargs)
        evidence = pending.pop((kwargs['frame'], kwargs['view']), None)
        if evidence is not None:
            refs, posterior = evidence
            ids = result[0][-1].track_ids.tolist()
            for row, ref in enumerate(ids):
                if ref in refs: executor.memory.commitment.reliability[int(ref)] = float(posterior[row, refs.index(ref)])
        return result

    executor.scores = scores; executor.commit = commit
    return executor

"""GT-free, portable native asset primitives; no tracker/model imports."""
import hashlib
import json
import torch

INTERFACE = 'cvpil-dev12-native-assets/v1'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError('nonfinite native tensor')
    return hashlib.sha256(str((str(value.dtype), list(value.shape))).encode() + value.numpy().tobytes()).hexdigest()


def activate(logits, sizes):
    if logits.shape[1] != sum(sizes):
        raise ValueError('raw association column/block mismatch')
    parts = logits.split(sizes, dim=1)
    return torch.cat([torch.cat([p, p.new_zeros((len(p), 1))], 1).softmax(1)[:, :-1] for p in parts], 1)


def replay_call(call):
    """Reproduce the *joint* actual CPU Hungarian input/strict acceptance."""
    from scipy.optimize import linear_sum_assignment
    scores = call['track_scores']
    if tuple(scores.shape) != (len(call['queries']), len(call['track_ids'])):
        raise ValueError('joint matrix/metadata mismatch')
    tensor_digest(scores)
    rows, columns = linear_sum_assignment(-scores)
    accepted = [-1] * len(call['queries'])
    pairs = []
    for row, column in zip(rows, columns):
        score = scores[row, column]
        boundary = call['thresholds'][column]
        legal = bool(score > boundary)
        track = call['track_ids'][column]
        if legal:
            accepted[int(row)] = track
        pairs.append(dict(row=int(row), track=track, score=float(score), threshold=float(boundary), accepted=legal))
    return pairs, accepted


def validate_past(query, ref, completed_before):
    if ref['frame'] > query['frame']:
        raise ValueError('future reference')
    if ref['frame'] == query['frame'] and ref['source'][0] not in completed_before:
        raise ValueError('unprocessed same-group camera reference')


class CommitDictionary(dict):
    """Passive observer, preserving actual dict values and original set order."""
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.callback(int(key), value)

"""Persistent committed observations, with soft causal predecessor hypotheses."""
import copy
import torch
from torch.nn import functional as F

FEATURES = 20


class CommitmentState:
    VERSION = 1

    def __init__(self):
        self.observations = {}
        self.last_payload = None
        self.reliability = {}

    def state_dict(self):
        return copy.deepcopy(dict(version=self.VERSION, observations=self.observations,
                                  last_payload=self.last_payload, reliability=self.reliability))

    def load_state_dict(self, state):
        assert state['version'] == self.VERSION
        self.observations = copy.deepcopy(state['observations'])
        self.last_payload = state['last_payload']
        self.reliability = copy.deepcopy(state['reliability'])

    def update(self, ref, feature, box, image_size, frame, view):
        key = int(ref), int(view)
        old = self.observations.get(key)
        assert old is None or frame > old['frame'], 'duplicate same-camera ID or backwards time'
        visual = F.normalize(feature.detach(), dim=-1).clone()
        normalized = box.detach().clone() / box.new_tensor([image_size[1], image_size[0]] * 2)
        gap = frame - old['frame'] if old else None
        similarity = float(visual @ old['visual']) if old else 1.
        self.observations[key] = dict(frame=int(frame), box=normalized, visual=visual,
            previous_box=old['box'].clone() if old else None, previous_frame=old['frame'] if old else frame,
            run=old['run'] + 1 if old and gap == 1 else 1,
            hits=old['hits'] + 1 if old else 1,
            breaks=old['breaks'] + int(gap != 1) if old else 0,
            coherence=.9 * old['coherence'] + .1 * similarity if old else 1.,
            mean=.95 * old['mean'] + .05 * visual if old else visual.clone())
        self.last_payload = int(frame), int(view)

    def features(self, current, refs, frame, view):
        device = current.reid_features.device
        visual = F.normalize(current.reid_features[:, :1024], dim=-1)
        boxes = current.pred_boxes.tensor / current.pred_boxes.tensor.new_tensor(
            [current.image_size[1], current.image_size[0]] * 2)
        q, k = len(current), len(refs)
        out = visual.new_zeros(q, k, FEATURES)
        hypotheses = [(ref, item) for (ref, cam), item in self.observations.items()
                      if cam == view and 0 < frame - item['frame'] <= 8]
        weights = visual.new_zeros(q, len(hypotheses) + 1)
        weights[:, -1] = 4.  # An explicit unmatched hypothesis; never force a predecessor.
        positions = {}
        for col, (ref, item) in enumerate(hypotheses):
            gap = frame - item['frame']; predicted = item['box'].to(device)
            if item['previous_box'] is not None and item['frame'] > item['previous_frame']:
                predicted = predicted + (predicted - item['previous_box'].to(device)) / (
                    item['frame'] - item['previous_frame']) * gap
            weights[:, col] = 8 * (visual @ item['visual'].to(device)) - 20 * (
                boxes - predicted).abs().mean(-1) - .2 * gap
            positions[ref] = col
        weights = weights.softmax(-1)
        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(-1)
        for col, ref in enumerate(refs):
            own = self.observations.get((int(ref), int(view)))
            other = self.observations.get((int(ref), 1 - int(view)))
            support = weights[:, positions[ref]] if ref in positions else visual.new_zeros(q)
            out[:, col, 0] = support
            out[:, col, 1] = (support > weights[:, -1]).float()
            out[:, col, 2] = float(own is not None)
            out[:, col, 8] = entropy
            out[:, col, 9] = weights[:, -1]
            out[:, col, 10] = float(other is not None)
            if other:
                out[:, col, 11] = min(frame - other['frame'], 2000) / 40
                out[:, col, 12] = visual @ other['visual'].to(device)
            if own:
                gap = frame - own['frame']
                out[:, col, 3] = min(gap, 2000) / 40
                out[:, col, 4] = min(own['run'], 2000) / 40
                out[:, col, 5] = visual @ own['visual'].to(device)
                out[:, col, 6] = (boxes - own['box'].to(device)).abs().mean(-1)
                out[:, col, 7] = own['coherence']
                out[:, col, 13] = 1 - float(own['mean'].norm())
                out[:, col, 14] = own['breaks'] / own['hits']
                out[:, col, 18] = float(gap == 1)
                out[:, col, 19] = min(own['hits'], 2000) / 2000
            if q > 1:
                demand = support[None].expand(q, -1).clone()
                demand.fill_diagonal_(0)
                out[:, col, 15] = demand.max(-1).values
            reliability = self.reliability.get(int(ref))
            out[:, col, 16] = reliability if reliability is not None else .5
            out[:, col, 17] = float(reliability is not None)
        return out[None]

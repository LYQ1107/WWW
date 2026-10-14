"""Bounded causal target-to-clip retrieval. No identity labels enter this index."""
from collections import OrderedDict
import json
import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError('expected a finite nonzero target vector')
    return vector / np.linalg.norm(vector)


class CausalClipIndex:
    def __init__(self, clip_groups=30, ttl_groups=180, max_bytes=100 << 20, feature_age=20):
        if min(clip_groups, ttl_groups, max_bytes) < 1 or feature_age < 0:
            raise ValueError('invalid index bounds')
        self.clip_groups = clip_groups
        self.ttl_groups = ttl_groups
        self.max_bytes = max_bytes
        self.feature_age = feature_age
        self.entries = OrderedDict()
        self.last_round = -1
        self.bytes = 0
        self.peak_bytes = 0
        self.evictions = 0

    @staticmethod
    def entry_bytes(entry):
        # Includes arrays plus serialized public metadata. Python allocator/RSS
        # is reported separately by the live runner; this is a payload bound.
        return sum(entry[k].nbytes for k in ('gmt_sum', 'extra') if entry[k] is not None) + len(json.dumps(entry['metadata']).encode())

    def advance(self, round_id):
        if round_id < self.last_round:
            raise ValueError('index time cannot go backwards')
        self.last_round = round_id
        for key in list(self.entries):
            entry = self.entries[key]
            if round_id - entry['last_round'] > self.ttl_groups:
                self.bytes -= self.entry_bytes(entry)
                del self.entries[key]

    def publish(self, scene, camera, track, round_id, image_id, frame_id, bbox, gmt, extra=None):
        self.advance(round_id)
        gmt = unit(gmt)
        extra = None if extra is None else unit(extra)
        key = f'{int(scene)}:{int(camera)}:{int(track)}:{round_id // self.clip_groups}'
        entry = self.entries.pop(key, None)
        if entry is None:
            entry = dict(gmt_sum=np.zeros_like(gmt), extra=None, extra_round=None, count=0,
                         scene=int(scene), camera=int(camera), last_round=round_id,
                         metadata=dict(clip_id=key, scene=int(scene), camera=int(camera), track=int(track), observations=[]))
        else:
            self.bytes -= self.entry_bytes(entry)
        if entry['gmt_sum'].shape != gmt.shape:
            raise ValueError('target dimension changed within a clip')
        entry['gmt_sum'] += gmt
        entry['count'] += 1
        entry['last_round'] = round_id
        if extra is not None:
            entry['extra'] = extra
            entry['extra_round'] = round_id
        entry['metadata']['observations'].append(dict(image_id=int(image_id), frame_id=int(frame_id), round_id=int(round_id), bbox=list(map(float, bbox))))
        self.entries[key] = entry
        self.bytes += self.entry_bytes(entry)
        while self.bytes > self.max_bytes and self.entries:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= self.entry_bytes(removed)
            self.evictions += 1
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        return key

    def query(self, scene, camera, round_id, gmt, extra=None, feature_weight=0., scale=1., bias=0., top_k=10):
        self.advance(round_id)
        if top_k < 1:
            raise ValueError('top_k must be positive')
        gmt = unit(gmt)
        extra = None if extra is None else unit(extra)
        scores = []
        for key, entry in self.entries.items():
            if (entry['scene'], entry['camera']) == (scene, camera):
                continue
            score = float(gmt @ unit(entry['gmt_sum']))
            augmented = extra is not None and entry['extra'] is not None and round_id - entry['extra_round'] <= self.feature_age
            if augmented:
                score += feature_weight * float(np.tanh(scale * (extra @ entry['extra']) + bias))
            scores.append(dict(clip_id=key, score=score, used_extra=bool(augmented)))
        scores.sort(key=lambda row: (-row['score'], row['clip_id']))
        return dict(results=scores[:top_k], ranked_clip_ids=[row['clip_id'] for row in scores],
                    candidates=len(scores), augmented_candidates=sum(row['used_extra'] for row in scores))

    def snapshot(self):
        # Independent JSON value: later publishes cannot backfill old snapshots.
        return json.loads(json.dumps([entry['metadata'] for entry in self.entries.values()]))

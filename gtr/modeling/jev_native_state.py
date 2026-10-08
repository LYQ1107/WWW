"""Lossless native GMT prefix capture and fork through the production loop.

This module accepts no annotation or future-action map. Instances and their
ordered galleries remain actual production containers; torch serialization
preserves shared tensor storage instead of reconstructing vectors from hits.
"""
import copy
import dataclasses
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

STATE_VERSION = 'native_GMT_pre_association_prefix_v1'


def pack(value):
    if dataclasses.is_dataclass(value):
        return {field.name: pack(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if hasattr(value, 'get_fields'):
        return {'image_size': list(value.image_size), 'fields': {k: pack(v) for k, v in value.get_fields().items()}}
    if hasattr(value, 'tensor'):
        return value.tensor
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(v) for v in value]
    return value


def fingerprint(value):
    """Content hash includes every Instances field and ordered container."""
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            t = x.detach().cpu().contiguous()
            h.update(str((str(t.dtype), list(t.shape))).encode()); h.update(t.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((str(x.dtype), list(x.shape))).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=lambda k: (type(k).__name__, str(k))):
                h.update(str((type(k).__name__, k)).encode()); visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for item in x: visit(item)
        elif isinstance(x, set):
            visit(sorted(x))
        elif isinstance(x, np.generic):
            visit(x.item())
        elif x is None or isinstance(x, (bool, int, float, str)):
            h.update(json.dumps(x, allow_nan=False).encode() + b';')
        else:
            visit(pack(x))
    visit(pack(value))
    return h.hexdigest()


def prefix_state(model, *, instances, id_count, hits, galleries, frame, view,
                 first, frame_old_instances=()):
    from .meta_arch.gtr_rcnn import poss_ids, old_ids, old_reids
    rng = model._jev_trajectory_rng
    return {
        'version': STATE_VERSION, 'boundary': 'BEFORE_CURRENT_GET_ASSO_AND_MATCH',
        'key': [int(model._jev_context['video_id']), int(frame), int(view)],
        'first': bool(first), 'view_num': int(model._jev_context['view_num']),
        'instances': instances, 'id_count': int(id_count),
        'hits': hits, 'galleries': galleries,
        'possible_ids': poss_ids.poss_ids.copy(), 'old_ids': old_ids.old_ids.copy(),
        'old_reids': old_reids.old_reids,
        'frame_old_instances': frame_old_instances,
        'trajectory_rng_state': None if rng is None else rng.getstate(),
        'trajectory_rng_draws': None if rng is None else copy.deepcopy(rng._trajectory_draws),
        'context': copy.deepcopy(model._jev_context),
        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
    }


def restore_state(model, prefix):
    from .meta_arch.gtr_rcnn import poss_ids, old_ids, old_reids
    from .roi_heads.transformer import TrajectoryRandom
    if prefix['version'] != STATE_VERSION or prefix['boundary'] != 'BEFORE_CURRENT_GET_ASSO_AND_MATCH':
        raise ValueError('unsupported native prefix boundary/version')
    if model.jev_mode != 'off' or not model.jev_enabled:
        raise ValueError('native forks require enabled explicit trajectory RNG and frozen lifecycle MODE=off')
    poss_ids.poss_ids = prefix['possible_ids'].copy()
    old_ids.old_ids = prefix['old_ids'].copy()
    old_reids.old_reids = prefix['old_reids']
    if prefix['trajectory_rng_state'] is None:
        raise ValueError('native fork has no trajectory RNG provenance')
    model._jev_trajectory_rng = TrajectoryRandom()
    model._jev_trajectory_rng.setstate(prefix['trajectory_rng_state'])
    model._jev_trajectory_rng._trajectory_draws = copy.deepcopy(prefix['trajectory_rng_draws'])
    model._jev_context = copy.deepcopy(prefix['context'])
    random.setstate(prefix['python_rng']); np.random.set_state(prefix['numpy_rng'])
    torch.set_rng_state(prefix['torch_rng'].cpu())
    torch.cuda.set_rng_state(prefix['cuda_rng'].cpu())
    model._jev_candidate_new_rows = ()
    model.__dict__.pop('_jev_candidate_last', None)


class NativeProductionPrefixRecorder:
    """Observer before proposal and after commit; never changes a decision."""
    def __init__(self, model, wanted_keys, output, callback=None, resume=False):
        self.model = model; self.wanted = set(map(tuple, wanted_keys))
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.callback = callback; self.current = None; self.saved = {}; self.trace = []
        self.event_ledger = []

    def before(self, **values):
        key = (int(self.model._jev_context['video_id']), int(values['frame']), int(values['view']))
        start = 0 if values['first'] else max(0, values['frame'] + 1 - self.model.test_len) * values.get('view_num', 2)
        end = len(values['instances']) if values['first'] else values['frame'] * 2 + values['view']
        active = set()
        for instance in values['instances'][start:end]:
            if instance.has('track_ids'):
                active.update(map(int, instance.track_ids.detach().cpu().tolist()))
        self.current = {'key': key, 'id_count': values['id_count'], 'active_ids': active,
                        'lengths': {int(t): len(g) for t, g in values['galleries'].items()}}
        if key in self.wanted:
            p = self.output / ('PREFIX_%02d_%06d_%d.pth' % key)
            current = prefix_state(self.model, **values)
            if p.exists():
                if not self.resume:
                    raise FileExistsError('prefix output is immutable: ' + str(p))
                assert fingerprint(torch.load(p, map_location=self.model.device)) == fingerprint(current), 'resumed prefix differs'
            else:
                tmp = p.with_suffix('.pth.tmp')
                torch.save(current, tmp); tmp.replace(p)
            self.saved[key] = {'key': list(key), 'path': str(p)}

    def after(self, **values):
        candidate = values.pop('candidate'); rng = values.pop('trajectory_rng')
        key = (int(self.model._jev_context['video_id']), int(values['frame']), int(values['view']))
        before = self.current
        assert before is not None and before['key'] == key
        actual = values['instances'][-1]
        events = []
        for row, track in enumerate(actual.track_ids.detach().cpu().tolist()):
            track = int(track); before_len = before['lengths'].get(track, 0)
            after_len = len(values['galleries'][track])
            if track > before['id_count']:
                actions = ['BIRTH_INITIALIZE', 'NEW_ID_COMMIT']
            else:
                bank = track not in before['active_ids']
                actions = ['BANK_REACTIVATE'] if bank else ['EXISTING_MATCH']
                actions += ['REACTIVATION_WRITE' if bank else 'MEMORY_WRITE'] if after_len > before_len else ['MEMORY_SKIP']
            events.append({'key': list(key), 'row': row, 'track': track,
                           'events': actions, 'gallery_before': before_len,
                           'gallery_after': after_len,
                           'vector_source': {'cache_key': list(key), 'row': row},
                           'actual_vector_SHA256': fingerprint(actual.reid_features[row]),
                           'actual_hits': int(values['hits'][track])})
        ledger_sha = fingerprint(events)
        row = {'key': list(key), 'ids': actual.track_ids.detach().cpu().tolist(),
               'id_count': int(values['id_count']), 'event_SHA256': ledger_sha,
               'full_native_commit_SHA256': fingerprint(dict(values, trajectory_rng_state=rng.getstate(), trajectory_rng_draws=rng._trajectory_draws)),
               'events': events}
        self.trace.append(row)
        if key in self.wanted:
            packet = None if candidate is None else {
                'candidate_ids': candidate['candidate_ids'],
                'batch': candidate['batch'], 'assignment': candidate['assignment']}
            path = self.output / ('PACKET_%02d_%06d_%d.pth' % key)
            packet_result = {'packet': packet, 'commit': row}
            if path.exists():
                assert self.resume and fingerprint(torch.load(path, map_location=self.model.device)) == fingerprint(packet_result), 'resumed packet differs'
            else:
                tmp = path.with_suffix('.pth.tmp')
                torch.save(packet_result, tmp); tmp.replace(path)
            self.saved[key]['packet_path'] = str(path)
        if self.callback is not None:
            self.callback(row, self.saved)


class NativeStateForkAdapter:
    """Replay is the actual resumed production loop, not the old memory mirror."""
    def __init__(self, model):
        self.model = model

    def run(self, prefix_path, inputs, *, stop_frame, raw=True):
        prefix = torch.load(prefix_path, map_location=self.model.device)
        return self.model.sliding_inference_GMT(inputs, prefix['view_num'],
            [None, None, list(range(len(inputs)))], native_prefix=prefix,
            native_stop_frame=stop_frame, native_raw=raw)

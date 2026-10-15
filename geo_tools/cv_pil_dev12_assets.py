"""Strict read-only consumer for cvpil-dev12-native-assets/v1.

This is a canonical semantic mapping, not a required producer file layout.
Inputs, features, actions, original call anchors, and OFFLINE labels occupy
different SHA-bound files. No tracker, GT matcher, collector or teacher is run.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment

from cv_pil_native_handoff import (FROZEN_STAGE2_SHA256, FROZEN_CONFIG_SHA256,
                                   sha256, valid_sha, integer, safe_torch_load)

SCHEMA = 'cvpil-dev12-native-assets/v1'
ROLES = {'inputs', 'features', 'actions', 'native_context', 'offline_supervision'}
SCORE_CONTRACT = 'per_reference_block_softmax_with_zero_unmatched_then_sum_by_global_id'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def exact(obj, names, where):
    require(isinstance(obj, dict) and set(obj) == set(names), 'Illegal fields in ' + where)


def key(value):
    require(isinstance(value, list) and len(value) == 3 and
            all(integer(x) and x >= 0 for x in value) and value[0] == 12,
            'Invalid DEV12 physical detection key')
    return tuple(value)


def vector(value):
    require(isinstance(value, torch.Tensor) and value.shape == (1152,) and
            value.is_floating_point() and torch.isfinite(value).all().item() and
            value.float().norm().item() > 0, 'Missing or invalid full 1152 feature')
    return value.detach().float().clone()


class Dev12Assets:
    """Validated A/B data only. Labels are loaded by a separate explicit method."""
    def __init__(self, manifest_path, expected_sha256, expected_provenance, authorized_root,
                 expected_scope):
        path = Path(manifest_path).resolve()
        root = Path(authorized_root).resolve()
        require(path.is_relative_to(root), 'Manifest outside authorized root')
        require(valid_sha(expected_sha256) and sha256(path) == expected_sha256,
                'Manifest SHA mismatch')
        self.manifest = json.loads(path.read_text())
        m = self.manifest
        require(m.get('schema') == SCHEMA and m.get('status') == 'FROZEN_COMPLETE',
                'Incomplete or incompatible release')
        require(m.get('provenance') == expected_provenance, 'Independent source binding mismatch')
        p = expected_provenance
        require(p.get('checkpoint_sha256') == FROZEN_STAGE2_SHA256 and
                p.get('config_sha256') == FROZEN_CONFIG_SHA256 and
                re.fullmatch('[0-9a-f]{40}', p.get('source_commit', '')) is not None and
                all(valid_sha(p.get(x)) for x in ('video_sha256', 'camera_map_sha256', 'gt_sha256')),
                'Original GMT/data provenance required')
        scope = m.get('scope', {})
        require(scope == expected_scope, 'Independent frozen scope binding mismatch')
        require(scope.get('video') == 12 and scope.get('selection_uses_gt') is False and
                scope.get('all_primary_queries') is True and
                scope.get('synchronized_groups') == 1200 and
                scope.get('frozen_before_export') is True and
                valid_sha(scope.get('preregistration_sha256')) and
                valid_sha(scope.get('query_keys_sha256')), 'Unfrozen or selective DEV scope')
        self.paths = {}
        require(set(m.get('artifacts', {})) == ROLES, 'Missing semantic artifact roles')
        for role, item in m['artifacts'].items():
            file = (path.parent / item['path']).resolve()
            require(file.is_relative_to(root) and file.is_file(), 'Artifact outside authorized root')
            require(valid_sha(item.get('sha256')) and sha256(file) == item['sha256'],
                    'Artifact SHA mismatch: ' + role)
            self.paths[role] = file
        require(len(set(self.paths.values())) == len(ROLES), 'GT must be physically isolated')
        # weights_only avoids executable pickle objects; all other tables are JSON.
        self.features = safe_torch_load(self.paths['features'])
        self.inputs = json.loads(self.paths['inputs'].read_text())
        self.actions = json.loads(self.paths['actions'].read_text())
        self.context = json.loads(self.paths['native_context'].read_text())
        self._validate()

    def _validate(self):
        exact(self.inputs, ('observations', 'calls'), 'input tables (GT forbidden)')
        self.obs = {}
        for o in self.inputs['observations']:
            exact(o, ('key', 'frame', 'camera', 'causal_index', 'feature_ref'), 'observation')
            k = key(o['key'])
            require(k not in self.obs, 'Duplicate physical observation')
            require(integer(o['frame']) and o['frame'] >= 1 and integer(o['camera']) and
                    o['camera'] in (1, 2) and integer(o['causal_index']) and o['causal_index'] >= 0,
                    'Invalid physical camera/frame/causal clock')
            require(o['feature_ref'] in self.features, 'Missing native history feature')
            vector(self.features[o['feature_ref']])
            self.obs[k] = o
        require(len({o['feature_ref'] for o in self.obs.values()}) == len(self.obs),
                'Different physical observations cannot share a feature reference')
        require(set(self.features) == {o['feature_ref'] for o in self.obs.values()},
                'Unexpected data in feature table')
        self.calls = {}
        self.query_index = {}
        require(set(self.context) == set(self.actions), 'Native action/context call mismatch')
        for c in self.inputs['calls']:
            exact(c, ('call_id', 'query_keys', 'candidate_ids', 'track_scores', 'solver_matrix',
                      'reference_keys', 'reference_candidate_ids', 'reference_block_sizes',
                      'logits', 'histories', 'score_contract'), 'native call')
            cid = c['call_id']
            require(isinstance(cid, str) and cid not in self.calls and cid in self.context,
                    'Missing/duplicate full native context')
            ids = c['candidate_ids']; queries = [key(k) for k in c['query_keys']]
            require(all(integer(i) and i >= 0 for i in ids) and len(ids) == len(set(ids)) and
                    len(queries) == len(set(queries)) and len(queries) > 0, 'Candidate/row IDs invalid')
            anchor = self.context[cid]
            exact(anchor, ('query_keys', 'candidate_ids', 'solver_matrix_sha256', 'action_sha256',
                           'observation_metadata_sha256'),
                  'original native call anchor')
            # The anchor is a separately sealed original trace, not a self-declared
            # "full=True" flag on an exported Top8 submatrix.
            require(anchor['query_keys'] == c['query_keys'] and anchor['candidate_ids'] == ids and
                    anchor['solver_matrix_sha256'] == digest_json(c['solver_matrix']),
                    'Top8/local rows or columns cannot impersonate full Hungarian context')
            n, m = len(queries), len(ids)
            require(all(isinstance(c[f], list) and len(c[f]) == n and
                        all(isinstance(r, list) and len(r) == m for r in c[f])
                        for f in ('track_scores', 'solver_matrix')), 'Full native matrix shape differs')
            score = torch.tensor(c['track_scores'], dtype=torch.float32).reshape(n, m)
            matrix = torch.tensor(c['solver_matrix'], dtype=torch.float32).reshape(n, m)
            require(torch.isfinite(score).all().item() and torch.equal(score, matrix),
                    'Native score/solver column mismatch')
            refs = [key(k) for k in c['reference_keys']]
            rid = c['reference_candidate_ids']
            require(len(refs) == len(rid) and len(refs) == len(set(refs)) and
                    set(rid) == set(ids) and all(k in self.obs for k in refs + queries),
                    'Native references/candidate ownership mismatch')
            require(digest_json([self.obs[k] for k in refs + queries]) ==
                    anchor['observation_metadata_sha256'],
                    'Original physical camera/frame/causal source binding differs')
            blocks = c['reference_block_sizes']
            require(all(integer(b) and b > 0 for b in blocks) and sum(blocks) == len(refs),
                    'Native reference block layout invalid')
            require(c['score_contract'] == SCORE_CONTRACT, 'Unrecognized native score normalization')
            require(len(c['logits']) == n and all(len(r) == len(refs) for r in c['logits']),
                    'Native logit row/column layout differs')
            logits = torch.tensor(c['logits'], dtype=torch.float32).reshape(n, len(refs))
            require(torch.isfinite(logits).all().item(), 'Invalid native logits')
            activated = [torch.cat((block, block.new_zeros((n, 1))), 1).softmax(1)[:, :-1]
                         for block in logits.split(blocks, dim=1)]
            prob = torch.cat(activated, 1) if activated else logits
            incidence = torch.tensor([[int(a == b) for b in ids] for a in rid],
                                     dtype=torch.float32).reshape(len(refs), m)
            require(torch.allclose(prob @ incidence, score, atol=3e-5, rtol=4e-6),
                    'Original GTA normalization/aggregation differs')
            histories = c['histories']
            require([h['global_id'] for h in histories] == ids, 'History candidate ID/column alignment')
            for h in histories:
                exact(h, ('global_id', 'sources', 'observed_length', 'support_count'), 'history')
                expected = [list(k) for k, i in zip(refs, rid) if i == h['global_id']]
                require(h['sources'] == expected and h['support_count'] == len(expected) and
                        integer(h['observed_length']) and h['observed_length'] >= len(expected),
                        'Missing/truncated native history or false support count')
            for row, q in enumerate(queries):
                require(q not in self.query_index, 'Duplicate query across calls')
                qo = self.obs[q]
                for ref in refs:
                    ro = self.obs[ref]
                    require(ro['causal_index'] < qo['causal_index'] and ro['frame'] <= qo['frame'],
                            'Noncausal/future native history')
                self.query_index[q] = (cid, row)
            self._validate_action(c, anchor, n, m)
            self.calls[cid] = c
        require(set(self.calls) == set(self.context), 'Missing full native calls')
        scope = self.manifest['scope']
        ordered_keys = [list(k) for k in self.query_index]
        require(len(ordered_keys) == scope.get('query_count') and
                digest_json(ordered_keys) == scope['query_keys_sha256'], 'Frozen scope coverage differs')

    def _validate_action(self, c, anchor, n, m):
        a = self.actions[c['call_id']]
        exact(a, ('pre_state_sha256', 'post_state_sha256', 'hungarian', 'thresholds',
                  'accepted_ids', 'bank_ids', 'birth_ids', 'final_ids', 'gallery_update_sha256',
                  'bank_update_sha256'), 'native action')
        require(all(valid_sha(a[x]) for x in ('pre_state_sha256', 'post_state_sha256',
                                             'gallery_update_sha256', 'bank_update_sha256')),
                'Missing native state/update evidence')
        require(digest_json(a) == anchor['action_sha256'], 'Native action trace binding differs')
        for field in ('thresholds', 'accepted_ids', 'bank_ids', 'birth_ids', 'final_ids'):
            require(len(a[field]) == (m if field == 'thresholds' else n), 'Action row/column mismatch')
        require(all(isinstance(x, (float, int)) and not isinstance(x, bool) and
                    torch.isfinite(torch.tensor(float(x))).item() and x >= 0 for x in a['thresholds']),
                'Invalid native accept/reject thresholds')
        rows, cols = [], []
        expected = [None] * n
        for pair in a['hungarian']:
            require(len(pair) == 2 and all(integer(i) for i in pair), 'Invalid Hungarian pair')
            r, col = pair
            require(0 <= r < n and 0 <= col < m, 'Invalid Hungarian row/column')
            rows.append(r); cols.append(col)
            if c['track_scores'][r][col] > a['thresholds'][col]:
                expected[r] = c['candidate_ids'][col]
        require(len(rows) == len(set(rows)) and len(cols) == len(set(cols)) and
                len(rows) == min(n, m) and expected == a['accepted_ids'],
                'Hungarian/threshold accepted action mismatch')
        matrix = torch.tensor(c['solver_matrix'], dtype=torch.float32).reshape(n, m)
        optimal_rows, optimal_cols = linear_sum_assignment(-matrix.numpy())
        require(torch.isclose(matrix[rows, cols].sum(),
                              matrix[optimal_rows, optimal_cols].sum(), atol=1e-6, rtol=1e-6).item(),
                'Recorded native assignment is not an optimal full Hungarian solve')
        # The original trace is the authority. Do not synthesize Bank state or
        # newborn GT identities from scalar rank predictions.
        for r in range(n):
            accepted, bank, birth, final = (a[f][r] for f in
                                           ('accepted_ids', 'bank_ids', 'birth_ids', 'final_ids'))
            require(all(x is None or (integer(x) and x >= 0) for x in (accepted, bank, birth, final))
                    and final is not None, 'Invalid native committed ID')
            require(sum(x is not None for x in (accepted, bank, birth)) == 1 and
                    final == next(x for x in (accepted, bank, birth) if x is not None),
                    'Bank/birth/final commit semantics differ')

    def actor(self, physical_key):
        """Exact legacy model schema; every full legal candidate is retained.

        Same-scene-time references remain in original solver context. Only
        strictly earlier scene frames enter the frozen old heads (lag > 0).
        All such references are used, with no Top8/last4 truncation.
        """
        q = key(list(physical_key)); cid, row = self.query_index[q]
        c = self.calls[cid]; o = self.obs[q]; histories = []
        for h in c['histories']:
            past = []
            for source in h['sources']:
                r = self.obs[tuple(source)]
                if r['frame'] == o['frame']:
                    continue
                past.append(dict(camera=r['camera'], lag=o['frame'] - r['frame'],
                                 feature=vector(self.features[r['feature_ref']]),
                                 cross_view=r['camera'] != o['camera']))
            histories.append(past)
        return dict(event_key=q, video=12, frame=o['frame'], camera=o['camera'],
                    query_feature=vector(self.features[o['feature_ref']]),
                    native_candidate_ids=tuple(c['candidate_ids']),
                    native_scores=torch.tensor(c['track_scores'][row], dtype=torch.float32),
                    past_histories=histories)

    def supervision(self):
        """Explicit offline read, never called by actor or input normalization."""
        require(sha256(self.paths['offline_supervision']) ==
                self.manifest['artifacts']['offline_supervision']['sha256'], 'Offline SHA changed')
        labels = json.loads(self.paths['offline_supervision'].read_text())
        require(len(labels) == len(self.query_index), 'Offline coverage differs')
        result = {}
        for s in labels:
            exact(s, ('key', 'GT_person', 'candidate_persons', 'primary_state',
                      'unknown_reason', 'impure_candidate_ids', 'reachable_positive_ids'), 'offline label')
            k = key(s['key']); require(k in self.query_index and k not in result, 'Illegal labeled query')
            c = self.calls[self.query_index[k][0]]; ids = c['candidate_ids']
            require(len(s['candidate_persons']) == len(ids) and
                    set(s['impure_candidate_ids']) <= set(ids) and
                    set(s['reachable_positive_ids']) <= set(ids), 'GT cannot add legal candidates')
            gt = s['GT_person']; persons = s['candidate_persons']
            require(gt is None or (integer(gt) and gt >= 0), 'Invalid GT identity')
            require(all(x is None or (integer(x) and x >= 0) for x in persons), 'Invalid candidate label')
            known = tuple(gt is not None and x is not None and i not in s['impure_candidate_ids']
                          for i, x in zip(ids, persons))
            positives = tuple(i for i, x in enumerate(persons) if known[i] and x == gt)
            require(s['reachable_positive_ids'] == [ids[i] for i in positives], 'Alias/reachable labels differ')
            require(s['primary_state'] in ('correct_link', 'wrong_link', 'missed_known_positive',
                                           'unknown', 'correct_reject', 'candidate_missing'),
                    'Unrecognized offline primary certification')
            cid, row = self.query_index[k]
            accepted = self.actions[cid]['accepted_ids'][row]
            index = ids.index(accepted) if accepted is not None else None
            if s['primary_state'] == 'wrong_link':
                require(index is not None and known[index] and index not in positives,
                        'Uncertified primary wrong-link label')
            if s['primary_state'] == 'correct_link':
                require(index in positives, 'Uncertified primary correct-link label')
            if s['primary_state'] == 'missed_known_positive':
                require(accepted is None and bool(positives), 'Uncertified missed-positive label')
            result[k] = dict(s, certified_target_indices=positives, candidate_known_mask=known,
                             no_legal_correct_candidate=not positives)
        return result


def off_passthrough(original_callable, *args, **kwargs):
    """OFF boundary: invoke original unchanged, return exact original object.

    Unit testing this boundary is not a full-video GMT parity certificate.
    Real online parity remains a separately gated full-native measurement.
    """
    return original_callable(*args, **kwargs)


def verify_frozen_evidence(manifest):
    records = manifest['checkpoints'] + [a for r in manifest['repositories'] for a in r['artifacts']]
    for item in records:
        p = Path(item['path'])
        require(p.stat().st_size == item['bytes'] and sha256(p) == item['sha256'],
                'Frozen old evidence changed: ' + str(p))
    return len(records)

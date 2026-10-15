"""CV-PIL-owned semantic adapter for the producer's actual native shard layout.

No imports from producer research code. Role paths are caller-supplied; filenames
are not an admission criterion. Prefix40 is engineering-only, never evaluation.
"""
from collections import Counter
import gzip
import json
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment

from cv_pil_dev12_assets import SCHEMA, require, vector, digest_json
from cv_pil_native_handoff import (sha256, safe_torch_load, FROZEN_STAGE2_SHA256,
                                   FROZEN_CONFIG_SHA256)


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def tensor_digest(t):
    t = t.detach().cpu().contiguous()
    import hashlib
    return hashlib.sha256(str((str(t.dtype), list(t.shape))).encode() + t.numpy().tobytes()).hexdigest()


class NativeShardAssets:
    def __init__(self, manifest_path, manifest_sha, scope_sha, provenance_pins, roles,
                 seal_path=None, seal_sha=None, prefix_engineering_only=False):
        self.root = Path(manifest_path).resolve().parent
        require(sha256(manifest_path) == manifest_sha, 'Manifest SHA mismatch')
        self.manifest = json.loads(Path(manifest_path).read_text()); m = self.manifest
        require(m.get('interface_version') == SCHEMA, 'Unsupported native shard semantics')
        require(m['provenance']['checkpoint_sha256'] == FROZEN_STAGE2_SHA256 and
                m['provenance']['config_sha256'] == FROZEN_CONFIG_SHA256,
                'Original GMT checkpoint/config required')
        require(all(m['provenance'].get(k) == v for k, v in provenance_pins.items()), 'Source provenance differs')
        require(m.get('scope_sha256') == scope_sha, 'External scope SHA differs')
        self.engineering_only = prefix_engineering_only
        if prefix_engineering_only:
            require(m['status'] == 'CAPTURE_COMPLETE_PARITY_PASS_UNLABELED' and m['groups'] == 40,
                    'Only frozen40 prefix is admitted for engineering')
        else:
            require(m['status'] == 'SEALED_ASSET_READY' and m['groups'] == 1200 and
                    m.get('offline_labels_separate') is True, 'Full sealed DEV12 release required')
            require(seal_path is not None and seal_sha is not None and sha256(seal_path) == seal_sha,
                    'Independent final seal SHA required')
            seal = json.loads(Path(seal_path).read_text())
            require(seal['status'] == 'SEALED_ASSET_READY', 'Incomplete external seal')
            for name, digest in seal['artifacts'].items(): self._checked(name, digest)
        for name, digest in m['artifacts'].items(): self._checked(name, digest)
        self.roles = roles
        self.observations = {}; self.variants = safe_torch_load(self._role('variants'))
        for name in roles['observations']:
            body = safe_torch_load(self._checked(name, m['artifacts'][name]))
            for source, obs in body.items():
                require(source not in self.observations and list(source) == obs['source'], 'Duplicate/source mismatch')
                require(obs['camera'] in (1, 2) and not isinstance(obs['camera'], bool), 'Physical camera invalid')
                require(obs['feature'].dtype == torch.float32 and
                        tensor_digest(obs['feature']) == obs['feature_sha256'], 'Native FP32 vector changed')
                vector(obs['feature']); self.observations[source] = obs
        self.calls = {}; self.query_index = {}; self.max_aggregation_error = 0.
        for name in roles['calls']:
            for call in safe_torch_load(self._checked(name, m['artifacts'][name])):
                cid = call['call_id']; require(cid not in self.calls, 'Duplicate call ID')
                self._validate_call(call); self.calls[cid] = call
                if call['stage'] != 'run_memory_tracker':
                    for row, source in enumerate(call['queries']):
                        k = (12, *source); require(k not in self.query_index, 'Duplicate physical query')
                        self.query_index[k] = (cid, row)
        stream = list(rows(self._role('queries')))
        require(len(stream) == len(self.query_index) == m['native_queries'], 'Missing complete query rows')
        seen = set()
        for row in stream:
            k = tuple(row['key']); require(k not in seen and k in self.query_index, 'Query stream key mismatch')
            seen.add(k); cid, qi = self.query_index[k]; call = self.calls[cid]
            require((cid, qi) == (row['call_id'], row['query_row']) and
                    row['final_track'] == call['decision']['final_track_ids'][qi] and
                    row['primary_accepted_track'] == call['decision']['accepted_track_ids'][qi],
                    'Original action/query linkage mismatch')
        require(len(self.calls) == m['total_calls'] and len(self.observations) == m['native_observations'],
                'Native manifest coverage mismatch')
        self._validate_commits()

    def _checked(self, name, digest):
        p = (self.root / name).resolve()
        require(p.is_relative_to(self.root) and p.is_file() and sha256(p) == digest,
                'Native artifact SHA/path mismatch: ' + name)
        return p

    def _role(self, role):
        name = self.roles[role]
        require(name in self.manifest['artifacts'], 'Missing semantic role: ' + role)
        return self._checked(name, self.manifest['artifacts'][name])

    def feature(self, spec):
        source = tuple(spec['source']); obs = self.observations[source]
        if spec['variant'] is None: return vector(obs['feature'])
        variant = self.variants[spec['variant']]
        require(variant['source'] == spec['source'] and
                tensor_digest(variant['feature']) == variant['feature_sha256'] and
                variant['feature'].dtype == torch.float32, 'Native descriptor version mismatch')
        return vector(variant['feature'])

    def _validate_call(self, c):
        q = c['queries']; refs = c['references']; ids = c['track_ids']; score = c['track_scores']
        require(ids == sorted(set(r['track'] for r in refs)) and
                score.shape == (len(q), len(ids)) and score.dtype == torch.float32 and
                torch.isfinite(score).all().item(), 'Full native candidate/row/score layout mismatch')
        require(len(c['query_feature_keys']) == len(q) and len(c['reference_feature_keys']) == len(refs) and
                sum(c['reference_block_sizes']) == len(refs), 'Missing full native features or reference blocks')
        support = Counter(r['track'] for r in refs)
        require(c['actual_candidate_support_lengths'] == [support[t] for t in ids], 'Native support count mismatch')
        require(all(c['observed_track_lengths'][str(t)] >= support[t] for t in ids) or
                c['stage'] == 'run_memory_tracker', 'Native history length inconsistent')
        raw = c['raw_logits']; sizes = c['activation_block_sizes']
        require(raw.shape == (len(q), len(refs)) and sum(sizes) == len(refs), 'Full native logit layout missing')
        parts = [torch.cat((b, b.new_zeros((len(q), 1))), 1).softmax(1)[:, :-1]
                 for b in raw.split(sizes, dim=1)]
        prob = torch.cat(parts, 1)
        saved = c['single_reference_probabilities']
        require(torch.allclose(prob, saved, atol=2e-6, rtol=2e-6), 'Native block activation differs')
        incidence = torch.tensor([[r['track'] == t for t in ids] for r in refs], dtype=torch.float32)
        rebuilt = saved @ incidence
        require(torch.allclose(rebuilt, score, atol=2e-5, rtol=2e-6), 'Native ID aggregation differs')
        if score.numel(): self.max_aggregation_error = max(self.max_aggregation_error, (rebuilt-score).abs().max().item())
        rr, cc = linear_sum_assignment(-score.numpy()); accepted = [-1] * len(q); pairs = []
        for r, col in zip(rr, cc):
            value = score[r, col]; boundary = c['thresholds'][col]; legal = bool(value > boundary)
            if legal: accepted[r] = ids[col]
            pairs.append(dict(row=int(r), track=ids[col], score=float(value), threshold=float(boundary), accepted=legal))
        require(pairs == c['decision']['hungarian_pairs'] and accepted == c['decision']['accepted_track_ids'],
                'Full joint native Hungarian/threshold action differs')
        for source, spec in zip(q, c['query_feature_keys']):
            require(source == spec['source'], 'Query feature physical source differs'); self.feature(spec)
        for r, spec in zip(refs, c['reference_feature_keys']):
            require(r['source'] == spec['source'], 'History feature physical source differs'); self.feature(spec)
            ro = self.observations[tuple(r['source'])]
            for source in q:
                qo = self.observations[tuple(source)]
                require(ro['frame'] < qo['frame'] or (ro['frame'] == qo['frame'] and
                        ro['image_id'] in c['completed_camera_images_before_call']), 'Noncausal native reference')
        if c['stage'] == 'run_memory_tracker':
            for track, support_spec in c['bank_descriptor_support'].items():
                require(support_spec is not None and support_spec['denominator'] > 0 and
                        len(support_spec['sources']) == support_spec['denominator'],
                        'Missing actual Bank descriptor support')
                total = 0
                for spec in support_spec['sources']:
                    total = total + self.feature(spec)
                    ro = self.observations[tuple(spec['source'])]
                    require(all(ro['frame'] < self.observations[tuple(source)]['frame'] or
                                (ro['frame'] == self.observations[tuple(source)]['frame'] and
                                 ro['image_id'] in c['completed_camera_images_before_call']) for source in q),
                            'Noncausal Bank support')
                total = total / support_spec['denominator']
                require(all(torch.allclose(self.feature(spec), total, atol=2e-6, rtol=2e-6)
                            for ref, spec in zip(refs, c['reference_feature_keys']) if ref['track'] == int(track)),
                        'Original native Bank average differs')

    def _validate_commits(self):
        previous = None; commits = {}; last = {}
        for i, row in enumerate(rows(self._role('commits'))):
            require(row['index'] == i and row['previous_commit_sha256'] == previous and
                    digest_json(row['state']) == row['state_sha256'], 'Native state chain differs')
            body = dict(row); previous = body.pop('commit_sha256')
            require(digest_json(body) == previous, 'Native commit digest differs')
            k = tuple(row['source']); require(k not in commits, 'Repeated physical commit')
            require(row['state']['reid_digests'][str(row['track'])] == row['reid_digest'], 'Gallery descriptor update differs')
            commits[k] = row['track']; last[row['call_id']] = row['state']
        require(set(commits) == set(self.observations), 'Physical observation/commit coverage differs')
        for c in self.calls.values():
            if c['stage'] == 'run_memory_tracker': continue
            require(not c['queries'] or last[c['call_id']] == c['state_after'], 'Post-state differs from last commit')
            require(all(commits[tuple(k)] == t for k, t in zip(c['queries'], c['decision']['final_track_ids'])),
                    'Final native submission linkage differs')

    def actor(self, k):
        cid, row = self.query_index[tuple(k)]; c = self.calls[cid]
        qo = self.observations[tuple(c['queries'][row])]; histories = []
        for track in c['track_ids']:
            past = []
            for ref, spec in zip(c['references'], c['reference_feature_keys']):
                ro = self.observations[tuple(ref['source'])]
                if ref['track'] != track or ro['frame'] == qo['frame']: continue
                past.append(dict(camera=ro['camera'], lag=qo['frame']-ro['frame'], feature=self.feature(spec),
                                 cross_view=ro['camera'] != qo['camera']))
            histories.append(past)
        return dict(event_key=tuple(k), video=12, frame=qo['frame'], camera=qo['camera'],
                    query_feature=self.feature(c['query_feature_keys'][row]),
                    native_candidate_ids=tuple(c['track_ids']), native_scores=c['track_scores'][row].clone(),
                    past_histories=histories)

    def supervision(self):
        require(not self.engineering_only, 'Prefix engineering assets are not an evaluation release')
        output = {}
        for s in rows(self._role('labels')):
            k = tuple(s['key']); require(k in self.query_index and k not in output, 'Offline physical key mismatch')
            cid, row = self.query_index[k]; c = self.calls[cid]; ids = c['track_ids']
            require((cid, row) == (s['call_id'], s['query_row']) and
                    s['final_track'] == c['decision']['final_track_ids'][row] and
                    s['primary_accepted_track'] == c['decision']['accepted_track_ids'][row],
                    'Offline label/native action mismatch')
            positives = set(s['positive_tracks']); ambiguous = set(s['ambiguous_tracks'])
            require(positives <= set(ids) and ambiguous <= set(ids) and not positives & ambiguous,
                    'GT cannot invent or disambiguate a legal candidate')
            gt = s['gt_identity']; pure = s['pure_track_gt']
            require(set(map(str, ids)) == set(pure) and
                    ambiguous == {t for t in ids if pure[str(t)] is None} and
                    positives == {t for t in ids if gt is not None and pure[str(t)] == gt} and
                    set(s['impure_tracks']) <= ambiguous, 'UNKNOWN/alias ownership labels conflict')
            known = tuple(s['gt_identity'] is not None and t not in ambiguous for t in ids)
            target = tuple(i for i, t in enumerate(ids) if t in positives)
            require(all(s['pure_track_gt'][str(t)] == s['gt_identity'] for t in positives), 'Alias certification differs')
            output[k] = dict(GT_person=s['gt_identity'], certified_target_indices=target,
                             candidate_known_mask=known, actual_primary_state=s['primary_state'].lower(),
                             final_native_commit_certified=False, source_label=s)
        require(set(output) == set(self.query_index), 'Offline label scope incomplete')
        return output

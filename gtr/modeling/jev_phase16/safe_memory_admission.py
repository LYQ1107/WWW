"""Past-only admission. Records point into immutable native raw Gallery rows.

CONFIRMED is a runtime consistency status, never a GT purity certificate.
No candidate IDs, commits, native hits, Bank or raw observations are removed.
"""
import copy
import torch
from torch.nn import functional as F

RULES = ['always_write', 'recent_only', 'conservative_provisional',
         'delayed_confirmation', 'bounded_quarantine']
COS = .85
BOX_L1 = .05
MAX_GAP = 2
CONFIRM_N = 3
PENDING_LIMIT = 4
QUARANTINE_LIMIT = 4


class SafeAdmission:
    def __init__(self, rule):
        assert rule in RULES
        self.rule = rule
        self.records = {}
        self.accepted = {}
        self.pending = {}
        self.quarantine = {}
        self.anchors = {}
        self.runs = {}
        self.transitions = []

    def state_dict(self):
        return copy.deepcopy(dict(version=1, rule=self.rule, records=self.records,
            accepted=self.accepted, pending=self.pending, quarantine=self.quarantine,
            anchors=self.anchors, runs=self.runs))

    def load_state_dict(self, state):
        assert state['version'] == 1 and state['rule'] == self.rule
        for key in ['records', 'accepted', 'pending', 'quarantine', 'anchors', 'runs']:
            setattr(self, key, copy.deepcopy(state[key]))
        self.transitions = []

    @staticmethod
    def norm(feature):
        return F.normalize(feature[:1024].detach().cpu().float(), dim=-1)

    def consistent(self, a, b, get_feature):
        sim = float(self.norm(get_feature(a['index'])) @ self.norm(get_feature(b['index'])))
        if sim < COS:
            return False
        if a['view'] == b['view'] and 0 < b['frame']-a['frame'] <= MAX_GAP:
            return float((b['box']-a['box']).abs().mean()) <= BOX_L1
        return True

    def append(self, identity, box, image_size, frame, view, get_feature, legacy=False):
        identity = int(identity)
        rows = self.records.setdefault(identity, [])
        index = len(rows)
        record = dict(index=index, frame=int(frame), view=int(view),
            box=box.detach().cpu().clone()/box.new_tensor([image_size[1], image_size[0]]*2).cpu(),
            status='PROVISIONAL', legacy=bool(legacy))
        rows.append(record)
        accepted = self.accepted.setdefault(identity, [])
        pending = self.pending.setdefault(identity, [])
        quarantine = self.quarantine.setdefault(identity, [])
        own = self.runs.get((identity, int(view)), [])
        contiguous = bool(own) and 0 < frame-rows[own[-1]]['frame'] <= MAX_GAP
        if not contiguous or not self.consistent(rows[own[-1]], record, get_feature):
            own = []
        own = (own+[index])[-CONFIRM_N:]
        self.runs[identity, int(view)] = own
        # Earliest stable same-camera triple. It is a consistency anchor only.
        if identity not in self.anchors and len(own) == CONFIRM_N:
            self.anchors[identity] = list(own)
        anchor = self.anchors.get(identity)
        if anchor:
            mean = torch.stack([self.norm(get_feature(j)) for j in anchor]).mean(0)
            compatible = float(self.norm(get_feature(index)) @ F.normalize(mean, dim=-1)) >= COS
        else:
            compatible = index == 0 or self.consistent(rows[index-1], record, get_feature)
        recent = next((rows[j] for j in reversed(accepted) if rows[j]['view'] == view), None)
        if recent and 0 < frame-recent['frame'] <= MAX_GAP:
            compatible = compatible and self.consistent(recent, record, get_feature)
        admitted = []
        if legacy or self.rule in ['always_write', 'recent_only']:
            admitted = [index]
        elif self.rule == 'conservative_provisional':
            if anchor and compatible:
                admitted = [index]
            else:
                pending.append(index)
        elif self.rule == 'bounded_quarantine' and accepted and anchor and compatible:
            admitted = [index]
        elif self.rule == 'bounded_quarantine' and not compatible:
            quarantine.append(index)
            record['status'] = 'QUARANTINED'
        else:
            # Delayed confirmation always needs three consistent observations.
            # Bounded quarantine needs this at cold start, without inventing evidence.
            if pending and not self.consistent(rows[pending[-1]], record, get_feature):
                for j in pending:
                    rows[j]['status'] = 'QUARANTINED'
                    quarantine.append(j)
                pending.clear()
            pending.append(index)
            if compatible and len(pending) >= CONFIRM_N:
                admitted = list(pending)
                pending.clear()
        for j in admitted:
            if j not in accepted:
                accepted.append(j)
            rows[j]['status'] = 'CONFIRMED_LEGACY_UNVALIDATED' if legacy else 'CONFIRMED'
        if self.rule == 'recent_only':
            accepted[:] = accepted[-4:]
        while len(pending) > PENDING_LIMIT:
            j = pending.pop(0)
            rows[j]['status'] = 'QUARANTINED'
            quarantine.append(j)
        quarantine[:] = quarantine[-QUARANTINE_LIMIT:]
        event = dict(identity=identity, index=index, frame=int(frame), view=int(view),
            status=record['status'], admitted_indices=admitted,
            anchor_available=bool(anchor), compatible_with_past=bool(compatible),
            pending_count=len(pending), quarantine_count=len(quarantine))
        self.transitions.append(event)
        return admitted, event

    def matching_indices(self, identity):
        # Explicit untrusted fallback keeps cold IDs lawful; never fabricate a token.
        accepted = self.accepted.get(int(identity), [])
        if accepted:
            return list(accepted), False
        return list(self.pending.get(int(identity), []))[-4:], True

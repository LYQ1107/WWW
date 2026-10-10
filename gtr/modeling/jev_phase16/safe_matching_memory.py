"""Independent trusted WHO/Q4 view; unconditional native raw state stays intact."""
import copy
import torch
from gtr.modeling.jev_phase15.native_commit_adapter import CommitmentMemory
from gtr.modeling.jev_phase16.safe_memory_admission import SafeAdmission
from gtr.modeling.jev_phase16.evidence_ledger import EvidenceLedger


class SafeMatchingMemory(CommitmentMemory):
    def __init__(self, rule, seed_prefix=None):
        super().__init__()
        self.rule = rule
        self.admission = SafeAdmission(rule)
        self.trusted = CommitmentMemory()
        self.raw_galleries = None
        self.seed_prefix = seed_prefix
        self.last_admission = []
        self.fallback_reads = 0

    def load_state_dict(self, state):
        if state['version'] == 16:
            super().load_state_dict(state['raw'])
            self.trusted.load_state_dict(state['trusted'])
            self.admission.load_state_dict(state['admission'])
            self.fallback_reads = state['fallback_reads']
            return
        super().load_state_dict(state)
        self.trusted.load_state_dict(state)
        assert self.seed_prefix is not None
        ledger = EvidenceLedger.from_prefix(self.seed_prefix)
        ledger.verify_raw_galleries(self.seed_prefix['galleries'])
        for identity, records in ledger.records.items():
            gallery = self.seed_prefix['galleries'][identity]
            for r in records:
                # All past writes shared identically; no GT retroactive filtering.
                self.admission.append(identity, r['box'], (1,1), *r['key'][:2],
                    lambda j, g=gallery: g.reid_features[j,:1024], legacy=True)
        self.admission.transitions.clear()
        self.seed_prefix = None

    def state_dict(self):
        return dict(version=16, rule=self.rule, raw=super().state_dict(),
            trusted=self.trusted.state_dict(), admission=self.admission.state_dict(),
            fallback_reads=self.fallback_reads)

    def update(self, ref, feature, box, image_size, frame, view):
        # This raw memory update remains exactly Phase XV's unconditional update.
        super().update(ref, feature, box, image_size, frame, view)
        assert self.raw_galleries is not None
        gallery = self.raw_galleries[int(ref)]
        indices, event = self.admission.append(ref, box, image_size, frame, view,
            lambda j: gallery.reid_features[j,:1024])
        assert len(self.admission.records[int(ref)]) == len(gallery)
        self.last_admission.append(event)
        if self.rule == 'recent_only':
            self.trusted.meta.pop(int(ref), None)
            self.trusted._history_cache.pop(int(ref), None)
            for key in [key for key in self.trusted.commitment.observations if key[0] == int(ref)]:
                del self.trusted.commitment.observations[key]
            indices = sorted(self.admission.accepted[int(ref)], key=lambda j:(self.admission.records[int(ref)][j]['frame'],self.admission.records[int(ref)][j]['view']))
        for j in indices:
            r = self.admission.records[int(ref)][j]
            old = self.trusted.commitment.observations.get((int(ref),r['view']))
            # Delayed promotion cannot backdate an already advanced trusted view.
            if old and r['frame'] <= old['frame']:
                self.admission.accepted[int(ref)].remove(j)
                r['status'] = 'QUARANTINED_BACKDATE'
                continue
            self.trusted.update(ref, gallery.reid_features[j,:1024], r['box'].to(gallery.reid_features.device), (1,1), r['frame'], r['view'])

    def build(self, current, refs, galleries, frame, view, active, **kwargs):
        self.raw_galleries = galleries
        self.commitment.last_payload = int(frame), int(view)
        if self.rule == 'always_write':
            return super().build(current, refs, galleries, frame, view, active, **kwargs)
        selected = {}
        for identity in refs:
            indices, fallback = self.admission.matching_indices(identity)
            self.fallback_reads += int(fallback)
            if not indices:
                # No admitted/pending real observation: last quarantined real row.
                indices = [len(galleries[identity])-1]
                self.fallback_reads += 1
                fallback = True
            full = indices == list(range(len(galleries[identity])))
            selected[identity] = galleries[identity] if full else galleries[identity][torch.tensor(indices,device=current.reid_features.device)]
        # Matching summaries and Q4 predecessor come from the independent view.
        self.trusted.commitment.reliability = copy.deepcopy(self.commitment.reliability)
        return self.trusted.build(current, refs, selected, frame, view, active, **kwargs)

"""TRAIN-only labels in a sidecar; never repair the predicted object bank."""
from collections import defaultdict
import numpy as np


class CausalMembershipLabels:
    def __init__(self):
        self.identities=defaultdict(set)
        self.anchors={}

    def targets(self,history,observations,labels):
        if len(observations)!=len(labels):raise ValueError("MEMBERSHIP_TARGET_SHAPE")
        # The first unambiguous labelled observation fixes the supervision
        # identity, not the runtime state. A later wrong reuse must remain a
        # negative example rather than removing this query from every loss.
        qlabels=[self.anchors.get(q.global_id) for q in history]
        positive=np.asarray([[a is not None and a==b for b in labels] for a in qlabels],bool).reshape(len(history),len(observations))
        return {"positive":positive,"history_valid":np.asarray([x is not None for x in qlabels],bool),
                "pure_history_valid":np.asarray([x is not None and len(self.identities[q.global_id])==1 for q,x in zip(history,qlabels)],bool),
                "observation_valid":np.asarray([x is not None for x in labels],bool),
                "labels":list(labels),"camera_slots":np.asarray([o.camera_id-1 for o in observations],np.int64),
                "history_labels":qlabels,"mixed_historical_objects":sum(len(self.identities[q.global_id])>1 for q in history)}

    def observe_after_decision(self,assignments,observations,labels):
        grouped=defaultdict(set)
        for o,label in zip(observations,labels):
            if label is not None:grouped[assignments[o.key]].add(label)
        for gid,values in grouped.items():
            self.identities[gid].update(values)
            if gid not in self.anchors:self.anchors[gid]=next(iter(values)) if len(values)==1 else None


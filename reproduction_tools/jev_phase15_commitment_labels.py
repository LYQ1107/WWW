"""TRAIN-only certificates; GT and label dictionaries never enter the actor."""
import torch


def certify(batch, refs, key, risk):
    assert risk.labels.video in [12, 13, 14, 16]
    frame, view = key; targets = risk.labels.current(frame, view)
    q, k = len(targets), len(refs)
    positive = torch.zeros(q, k+1, dtype=torch.bool)
    known = positive.clone(); availability = torch.full((q,), -1, dtype=torch.int8)
    trust = torch.full((k,), -1, dtype=torch.int8)
    safety = torch.full((q, k), -1, dtype=torch.int8)
    uncertainty = torch.full((q, k), -1, dtype=torch.int8)
    commit = positive.clone(); kind = torch.zeros(q, dtype=torch.int8)
    observed = []
    for col, ref in enumerate(refs):
        votes = risk.votes[ref]; n = sum(votes.values())
        coverage = n / max(1, risk.observations[ref])
        certified = n >= 3 and coverage >= .8
        pure = certified and len(votes) == 1
        polluted = certified and sum(value >= 2 for value in votes.values()) >= 2
        if pure: trust[col] = 1
        elif polluted: trust[col] = 0
        if pure or polluted: uncertainty[:, col] = int(polluted)
        observed.append((pure, polluted, dict(votes)))
    feature = batch['commitment_features'][0].detach().cpu()
    legal = batch['legal'][0].cpu()
    for row, gt in enumerate(targets):
        for col, (pure, polluted, votes) in enumerate(observed):
            known[row, col] = pure and gt is not None
            positive[row, col] = pure and gt is not None and gt in votes and legal[row, col]
            if known[row, col]: safety[row, col] = int(positive[row, col])
        if gt is None: continue
        if positive[row, :k].any(): availability[row] = 1; known[row, -1] = True
        elif known[row, :k].all():
            availability[row] = 0; positive[row, -1] = True; known[row, -1] = True
        previous = risk.latest.get((gt, view))
        if previous is not None and previous[1] in refs and frame - previous[0] <= 8:
            col = refs.index(previous[1])
            if positive[row, col] and feature[row, col, 0] > .5:
                commit[row, col] = True; kind[row] = 1  # Certified safe continuation.
        # A causal visual/motion predecessor confidently belonging to a different
        # certified person is unsafe to keep. Mixtures alone never prove this.
        if not kind[row] and k:
            col = int(feature[row, :, 0].argmax())
            if feature[row, col, 0] > .5 and known[row, col] and not positive[row, col] and positive[row, :k].any():
                commit[row] = positive[row]; kind[row] = 2
        if not kind[row] and availability[row] == 0:
            commit[row, -1] = True; kind[row] = 3  # Certified absence, native DEFER.
    supervised = (availability >= 0) | positive[:, :k].any(-1)
    targets_index = torch.full((q,), k, dtype=torch.long)
    for row in range(q):
        if positive[row].any(): targets_index[row] = int(torch.where(positive[row])[0][0])
    return dict(positive=positive, known_options=known, availability=availability, trust=trust,
                safety=safety, uncertainty=uncertainty, commit_positive=commit, commit_kind=kind,
                supervised=supervised, targets=targets_index, GT_labels_OFFLINE_ONLY=targets,
                commitment_UNKNOWN_rows=(kind == 0), multiple_pure_WHO_is_not_a_unique_commitment=True)

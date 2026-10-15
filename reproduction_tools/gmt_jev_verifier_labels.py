"""Offline certificates only. Never imported by a deployed verifier."""
import collections

def history_label(total,votes,owner,alias_count):
    known=sum(votes.values())
    if total<3 or known<3 or known/max(1,total)<.8:return 'UNKNOWN'
    substantial=[g for g,n in votes.items() if n>=3]
    if len(substantial)>=2:return 'CONFLICTED'
    if owner is not None and votes.get(owner,0)/total>=.8 and alias_count==1:return 'RELIABLE'
    return 'UNKNOWN'

def paired_plan(base,mlp,refs,targets,categories,people,owners):
    for plan in [base,mlp]:
        accepted=[c for c in plan if c>=0]
        assert len(accepted)==len(set(accepted)), 'Cannot splice illegal occupied identities'
    changed=[i for i,(a,b) in enumerate(zip(base,mlp)) if a!=b]
    if not changed:return dict(category='EQUIVALENT_NO_CHOICE',changed=[],label=None,people=[])
    rows=[];reasons=collections.Counter()
    for r in changed:
        a,b=base[r],mlp[r];target=targets[r];g=people[r]
        if a<0 or b<0:reasons['UNKNOWN_BANK_OR_BIRTH']+=1;continue
        if target<0:reasons[categories[r]]+=1;continue
        if owners.get(refs[a]) is None or owners.get(refs[b]) is None:
            reasons['UNKNOWN_ACCEPTED_OWNER']+=1;continue
        rows.append((a==target,b==target))
    if reasons:
        category='UNKNOWN_PLAN';label=None
    elif all(a for a,b in rows) and any(not b for a,b in rows):category='KEEP_GMT';label=0
    elif all(b for a,b in rows) and any(not a for a,b in rows):category='APPLY_MLP';label=1
    elif all(a and b for a,b in rows):category='BOTH_CORRECT_OR_EQUIVALENT';label=None
    elif not any(a or b for a,b in rows):category='BOTH_WRONG';label=None
    else:category='MIXED_PLAN_TRADEOFF_NO_LABEL';label=None
    return dict(category=category,changed=changed,label=label,people=sorted({people[r] for r in changed if people[r] is not None}),unknown_reasons=dict(reasons))

def dependency_components(events):
    """Person-connected plans are dependent even across cameras/time buckets."""
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for event in events:
        nodes=[(event['video'],p) for p in event['people']]
        for n in nodes[1:]:parent[find(n)]=find(nodes[0])
        if nodes:find(nodes[0])
    return len({find(x) for x in parent})

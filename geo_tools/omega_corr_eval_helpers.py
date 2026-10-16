"""Pure paired-evaluation contracts; unknown histories are never forced negatives."""
import numpy as np


def top_reference(scores,track_by_reference,native_rank):
    if not scores:return None
    order={t:i for i,t in enumerate(native_rank)}
    return min(scores,key=lambda r:(-scores[r],order.get(track_by_reference[r],10**9),r))


def certified_change(before,after):
    return dict(correction=before=='WRONG' and after=='CORRECT',harm=before=='CORRECT' and after=='WRONG',
                uncertified=('UNKNOWN' in before or 'UNKNOWN' in after or 'MIXED' in before or 'MIXED' in after))


def common_candidates(methods):
    return set.intersection(*(set(m) for m in methods)) if methods else set()


def cluster_interval(rows,seed=20261016,repeats=1000):
    # Rows: (scene, person, paired success difference). Sample whole scenes,
    # then scene-qualified people, carrying every correlated row for each person.
    groups={}
    for scene,person,value in rows:groups.setdefault(scene,{}).setdefault(person,[]).append(value)
    if len(groups)<2:return dict(status='INSUFFICIENT_SCENE_CLUSTERS',lower95=None,upper95=None,clusters=len(groups))
    rng=np.random.default_rng(seed);scenes=sorted(groups);samples=[]
    for _ in range(repeats):
        values=[]
        for scene in rng.choice(scenes,len(scenes),replace=True):
            people=sorted(groups[int(scene)])
            for person in rng.choice(people,len(people),replace=True):values.extend(groups[int(scene)][int(person)])
        samples.append(float(np.mean(values)))
    return dict(status='MEASURED',lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),
                mean=float(np.mean([r[2] for r in rows])),scene_clusters=len(groups),person_clusters=sum(map(len,groups.values())),replicates=repeats,seed=seed)

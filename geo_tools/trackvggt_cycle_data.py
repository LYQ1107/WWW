"""Actual-history cycle indices and separate post-parity visibility masks.

A(t) is the one selected detection plus NULL. A(t-1) and B(t) are
observation-only subsets of its actual Top8 predicted tracks. No GT-generated
history or future/after-current-camera bridge is accepted.
"""
import collections
from collect_track_hard_events import pure_reference_identity


def observable_cycle(call,event,observations,earlier_evidence):
    query=observations[tuple(event['query_source'])];now=query['frame'];camera=query['camera']
    if now<=1:return None,set()
    by_track=collections.defaultdict(list)
    for ref in call['references']:by_track[ref['track']].append(tuple(ref['source']))
    def ordered(keys):return sorted(keys,key=lambda key:(observations[key]['frame'],observations[key]['camera'],key))
    b=[]
    for track in event['candidate_tracks']:
        candidates=[s for s in by_track[track] if observations[s]['frame']==now and observations[s]['camera']!=camera and s in earlier_evidence]
        if not candidates:continue
        if len(candidates)!=1:raise ValueError('ambiguous current-camera source for a predicted identity')
        source=candidates[0]
        b.append(dict(track=track,source=list(source),history_sources=call['histories'][track]['sources'],observed_length=call['histories'][track]['observed_length']))
    if not b:return None,set()
    before=earlier_evidence[tuple(b[0]['source'])]['counts'];a=[]
    for track in event['candidate_tracks']:
        anchors=[s for s in by_track[track] if observations[s]['frame']==now-1 and observations[s]['camera']==camera]
        if not anchors or track not in before:continue
        if len(anchors)!=1:raise ValueError('ambiguous previous-camera source for a predicted identity')
        history=ordered([s for s in by_track[track] if observations[s]['frame']<now])[-4:]
        a.append(dict(track=track,source=list(anchors[0]),history_sources=[list(s) for s in history],observed_length=int(before[track])))
    if not a:return None,set()
    ab=[];maxima=[]
    for node in b:
        evidence=earlier_evidence[tuple(node['source'])];lookup={t:i for i,t in enumerate(evidence['track_ids'])}
        ab.append([float(evidence['scores'][lookup[row['track']]]) if row['track'] in lookup else None for row in a])
        maxima.append(float(evidence['scores'].max()) if len(evidence['scores']) else 0.)
    needed={tuple(source) for nodes in [a,b] for node in nodes for source in node['history_sources']}
    needed.update(tuple(node['source']) for nodes in [a,b] for node in nodes)
    if any(observations[s]['frame']>now for s in needed):raise ValueError('future cycle observation')
    graph=dict(previous_time=now-1,current_time=now,A_previous=a,B_current=b,A_current_sources=[event['query_source']],
        AB_native_scores_by_B_then_A=ab,B_native_score_normalizers=maxima,A_current_native_score_normalizer=max(event['native_track_scores']) if event['native_track_scores'] else 0.,
        same_actual_predicted_track_space=True,current_A_scope='one locked selected detection plus NULL',
        native_pair_scope='AB uses actual earlier B(t) solver-call evidence; BA/AA use actual current A(t) native evidence. Direct AA query history ends at t-1, while native A(t) evidence uses its full legal context.')
    return graph,needed


def label_cycle(graph,call,query_truth,labels,observation_metadata):
    refs=collections.defaultdict(list);past=collections.defaultdict(list)
    for ref in call['references']:
        source=tuple(ref['source']);value=labels[source];refs[ref['track']].append(value)
        if observation_metadata[source[0]]['frame_id']<=graph['previous_time']:past[ref['track']].append(value)
    full_pure={t:pure_reference_identity(v) for t,v in refs.items()};past_pure={t:pure_reference_identity(v) for t,v in past.items()}
    a_truth=[];b_truth=[]
    for row in graph['A_previous']:
        anchor=labels[tuple(row['source'])];identity=past_pure.get(row['track'])
        a_truth.append(identity if anchor is not None and anchor==identity else None)
    for node in graph['B_current']:
        observed=labels[tuple(node['source'])];identity=full_pure.get(node['track'])
        b_truth.append(identity if observed is not None and observed==identity else None)
    valid_b=[value is not None for value in b_truth]
    valid_ab=[[graph['AB_native_scores_by_B_then_A'][j][i] is not None and valid_b[j] and a_truth[i] is not None for j in range(len(b_truth))] for i in range(len(a_truth))]
    visible=[identity is not None and identity==query_truth and any(valid_ab[i][j] and b_truth[j]==identity for j in range(len(b_truth))) for i,identity in enumerate(a_truth)]
    return dict(visible_A_rows=visible,known_B_columns=valid_b,legal_AB_pairs=valid_ab,A_past_GT=a_truth,B_current_GT=b_truth,
        current_A_GT=query_truth,GT_used_only_for_visibility_and_loss_masks=True,
        scope='Pure actual full legal reference contexts, not purity of just the four summary observations. No unobserved bridge is completed with GT.')

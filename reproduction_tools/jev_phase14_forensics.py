"""Read saved native artifacts; never run an actor or infer labels from its scores."""
import argparse
import collections
import copy
import gzip
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from jev_phase14_common import *

CATEGORIES = ['WRONG_EXISTING_MATCH', 'FALSE_SPLIT', 'FALSE_DEFER', 'FALSE_MERGE',
              'WRONG_REACTIVATION', 'FALSE_BIRTH', 'GALLERY_CONTAMINATION',
              'IDENTITY_UNAVAILABLE', 'UNKNOWN_SUPERVISION', 'CROSS_CAMERA_ID_MISMATCH']

def load_gzip(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(line) for line in f]

def iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4).copy()
    b = np.asarray(b, dtype=float).reshape(-1, 4).copy()
    a[:, 2:] += a[:, :2]; b[:, 2:] += b[:, :2]
    lo = np.maximum(a[:, None, :2], b[None, :, :2]); hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi-lo, 0).prod(-1)
    area = np.maximum(a[:, 2:]-a[:, :2], 0).prod(-1)
    other = np.maximum(b[:, 2:]-b[:, :2], 0).prod(-1)
    return inter / np.maximum(area[:, None]+other[None]-inter, 1e-9)

def offline_alignment(raw, ann, video):
    images = {i['id']: i for i in ann['images'] if i['video_id'] == video}
    gt = collections.defaultdict(list); grouped = collections.defaultdict(list)
    for a in ann['annotations']:
        if a['image_id'] in images: gt[a['image_id']].append(a)
    for p in raw: grouped[p['image_id']].append(p)
    result = {}
    for image_id, im in images.items():
        ps = grouped[image_id]; targets = [None] * len(ps)
        if ps and gt[image_id]:
            scores = iou([p['bbox'] for p in ps], [a['bbox'] for a in gt[image_id]])
            rr, cc = linear_sum_assignment(-scores)
            for r, c in zip(rr, cc):
                if scores[r, c] >= .5: targets[r] = int(gt[image_id][c]['instance_id'])
        result[im['frame_id']-1, im['view_id']-1] = (ps, targets)
    return result

def native_choice(z, legal):
    z = np.asarray(z, dtype=np.float32).astype(np.float64)
    d, k1 = z.shape; k = k1-1; legal = np.asarray(legal, dtype=bool).reshape(d, k)
    cost = np.full((d, k+d), 1e9, dtype=float)
    cost[:, :k] = np.where(legal, -z[:, :k], 1e9)
    cost[np.arange(d), k+np.arange(d)] = -z[:, -1]
    rr, cc = linear_sum_assignment(cost)
    result = [-1] * d
    for r, c in zip(rr, cc): result[r] = int(c) if c < k else -1
    return result

def clear_switches(out, aligned):
    """Replay the exact saved TrackEval dataset/preprocessing/CLEAR scorer."""
    sys.path.insert(0, str(ROOT/'TrackEval'))
    for name, typ in [('float', float), ('int', int), ('bool', bool)]:
        if name not in np.__dict__: setattr(np, name, typ)
    import trackeval
    p = out/'strict_eval/tracking_eval_runtime_state/native/prepared/manifest.json'
    m = read(p)
    cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    cfg.update(GT_FOLDER=m['trackeval_gt'], TRACKERS_FOLDER=m['trackeval_trackers'],
               TRACKERS_TO_EVAL=['GMT'], BENCHMARK='VisionTrack', SPLIT_TO_EVAL='test',
               SKIP_SPLIT_FOL=True, SEQ_INFO=m['seq_lengths'], PRINT_CONFIG=False)
    dataset = trackeval.datasets.MotChallenge2DBox(cfg)
    events = []; counts = []
    for seq in m['sequences']:
        view = int(seq.rsplit('View', 1)[1])-1
        raw = dataset.get_raw_seq_data('GMT', seq)
        original_ids = [x.copy() for x in raw['tracker_ids']]
        original_boxes = [x.copy() for x in raw['tracker_dets']]
        data = dataset.get_preprocessed_seq_data(raw, 'pedestrian')
        metric = trackeval.metrics.CLEAR({'PRINT_CONFIG': False}).eval_sequence(data)
        prev = np.full(data['num_gt_ids'], np.nan); previous_step = prev.copy()
        before = len(events)
        for frame, (gs, ts, sim) in enumerate(zip(data['gt_ids'], data['tracker_ids'], data['similarity_scores'])):
            if not len(gs) or not len(ts): continue
            score = 1000*(ts[None, :] == previous_step[gs[:, None]]) + sim
            score[sim < .5-np.finfo(float).eps] = 0
            rr, cc = linear_sum_assignment(-score); valid = score[rr, cc] > np.finfo(float).eps
            rr = rr[valid]; cc = cc[valid]
            for r, c in zip(rr, cc):
                if not np.isnan(prev[gs[r]]) and prev[gs[r]] != ts[c]:
                    equal = np.all(np.isclose(original_boxes[frame], data['tracker_dets'][frame][c], rtol=0, atol=1e-9), axis=1)
                    possible = original_ids[frame][equal]; assert len(possible) == 1, (seq, frame, possible)
                    native = int(possible[0]); ps, targets = aligned[frame, view]
                    rows = [i for i, x in enumerate(ps) if x['track_id'] == native]
                    assert len(rows) == 1
                    events.append(dict(frame=frame, view=view, row=rows[0], native_id=native,
                                       clear_GT_relabelled=int(gs[r]), offline_GT=targets[rows[0]]))
            prev[gs[rr]] = ts[cc]; previous_step[:] = np.nan; previous_step[gs[rr]] = ts[cc]
        assert len(events)-before == metric['IDSW'], (seq, len(events)-before, metric['IDSW'])
        counts.append({'sequence': seq, 'reconstructed_IDSW': len(events)-before, 'official_IDSW': int(metric['IDSW'])})
    expected = read(out/'RESULT.json')['strict_online_metrics']['IDSW']
    assert len(events) == expected, (out, len(events), expected)
    return events, counts

def quantiles(values):
    return dict(zip(['min', 'p25', 'p50', 'p75', 'p95', 'max'], map(float, np.quantile(values, [0, .25, .5, .75, .95, 1])))) if values else None

def kbin(k):
    return '0' if k==0 else '1-9' if k<10 else '10-19' if k<20 else '20-29' if k<30 else '30-49' if k<50 else '50+'

def audit_case(out, ann, video, artifact_root=None):
    r = read(out/'RESULT.json'); rawpath = out/'RAW_PREDICTIONS.json'
    assert sha(rawpath) == r['raw_predictions']['SHA256']
    assert sha(out/'COMMITS.jsonl.gz') == r['commits_SHA256']
    assert sha(out/'QUESTIONS.jsonl.gz') == r['questions_SHA256']
    aligned = offline_alignment(read(rawpath), ann, video)
    commits = load_gzip(out/'COMMITS.jsonl.gz'); queries = collections.defaultdict(list)
    for q in load_gzip(out/'QUESTIONS.jsonl.gz'): queries[tuple(q['key'][1:])].append(q)
    switches, switch_counts = clear_switches(out, aligned)
    switch_keys = {(s['frame'], s['view'], s['row']): s for s in switches}
    votes = collections.defaultdict(collections.Counter); observations = collections.Counter()
    ever = set(); latest = {}; count = collections.Counter(); strata = collections.defaultdict(collections.Counter)
    margins = collections.defaultdict(list); examples = collections.defaultdict(list); trace = []
    defer_runs = collections.Counter(); repeat = collections.Counter(); episodes = collections.defaultdict(list)
    prior_view = 1-commits[0]['key'][2]
    ps, gts = aligned[0, prior_view]
    for p, gt in zip(ps, gts):
        observations[p['track_id']] += 1
        if gt is not None: votes[p['track_id']][gt] += 1; ever.add(gt); latest[gt, prior_view] = (0, p['track_id'])
    for event in commits:
        frame, view = event['key'][1:]; ps, gts = aligned[frame, view]
        assert event['ids'] == [p['track_id'] for p in ps]
        rowinfo = {}; initial = {}
        for q in queries[frame, view]:
            refs = q['refs']; choices = native_choice(q['logits'], q['legal']); task = q['task']
            for j, (row, col) in enumerate(zip(q['rows'], choices)):
                actual = event['events'][row]; expected_action = 'ASSOCIATE_EXISTING' if task==0 else 'REACTIVATE'
                assert (refs[col] if col>=0 else -1) == (actual['identity'] if actual['action']==expected_action else -1)
                gt = gts[row]; tags = []; support = [c for c, t in enumerate(refs) if gt is not None and set(votes[t]) == {gt}]
                known = [bool(votes[t]) and len(votes[t])==1 for t in refs]
                if task == 0: initial[row] = (col, support, refs)
                selected = 'DEFER' if col<0 else 'Selected UNKNOWN' if not known[col] else 'Certified Correct' if col in support else 'Certified Wrong'
                if gt is None: selected = 'Label Unavailable'; tags.append('UNKNOWN_SUPERVISION')
                unknown = [c for c in range(len(refs)) if not known[c]]
                stratum = strata[f'{"MATCH" if task==0 else "REACT"}/K={kbin(len(refs))}']
                stratum['rows'] += 1; stratum[selected] += 1
                if support: stratum['Correct Candidate Present'] += 1
                elif gt is not None:
                    stratum['Correct Candidate Missing'] += 1
                    if all(known): tags.append('IDENTITY_UNAVAILABLE'); stratum['Certified Unavailable'] += 1
                    else: stratum['Availability UNKNOWN'] += 1
                if gt is not None and not support and any(gt in votes[t] for t in refs): stratum['Contaminated Target Support'] += 1
                if task == 0 and gt is not None:
                    count['MATCH_'+selected] += 1
                    if selected == 'Certified Wrong': tags.append('WRONG_EXISTING_MATCH')
                    if support and col<0: tags.append('FALSE_DEFER'); stratum['False Defer'] += 1
                    z = np.asarray(q['logits'][j], dtype=np.float32)
                    if support and unknown:
                        margins['unknown_max_minus_correct_max'].append(float(z[unknown].max()-z[support].max()))
                        stratum['UNKNOWN_and_Correct_supported_rows'] += 1
                        if selected == 'Selected UNKNOWN':
                            stratum['UNKNOWN_steals_from_Correct'] += 1
                            count['UNKNOWN_selected_with_certified_correct_available'] += 1
                    if unknown:
                        margins['unknown_max_minus_terminal'].append(float(z[unknown].max()-z[-1]))
                        stratum['UNKNOWN_rows'] += 1
                        if col in unknown:
                            count['UNKNOWN_selected'] += 1
                            if gt is not None and votes[refs[col]] and any(g != gt for g in votes[refs[col]]):
                                count['UNKNOWN_selected_confirmed_polluted_prefix'] += 1
                            if gt is not None and gt not in votes[refs[col]] and votes[refs[col]]:
                                count['UNKNOWN_selected_new_confirmed_cross_GT_mix'] += 1
                    rowinfo.setdefault(row, {}).update(selected=selected, correct_available=bool(support), unknown_count=len(unknown), active_K=len(refs))
                for tag in tags:
                    count[tag] += 1
                    if len(examples[tag]) < 4: examples[tag].append(dict(key=[video, frame, view], row=row, gt=gt, candidate_refs=refs, selected=selected, pure_correct_refs=[refs[c] for c in support], provenance='past committed IoU>=0.5 observed GT only'))
        for row, (ref, gt, action) in enumerate(zip(event['ids'], gts, event['events'])):
            tags = []; before = dict(votes[ref]); seen = gt in ever if gt is not None else None
            was_polluted = any(len(v)>1 for v in votes.values())
            col, support, refs = initial[row]
            if gt is not None:
                if action['action']=='START_NEW' and seen: tags.append('FALSE_BIRTH')
                if action['action']=='START_NEW' and support: tags.append('FALSE_SPLIT')
                if action['action']=='REACTIVATE' and before and len(before)==1 and gt not in before: tags.append('WRONG_REACTIVATION')
                if before and gt not in before: tags += ['FALSE_MERGE', 'GALLERY_CONTAMINATION']
                if any(g != gt for g in before): count['writes_into_certified_mixed_history'] += 1
                previous = latest.get((gt, 1-view))
                if previous and frame-previous[0]<=40 and previous[1]!=ref: tags.append('CROSS_CAMERA_ID_MISMATCH')
                if col<0:
                    defer_runs[gt, view] += 1
                else:
                    if defer_runs[gt, view]: repeat['reassociation_after_DEFER_runs'] += 1
                    defer_runs[gt, view] = 0
                repeat['max_consecutive_DEFER_payloads'] = max(repeat['max_consecutive_DEFER_payloads'], defer_runs[gt, view])
                latest[gt, view] = (frame, ref)
            info = dict(key=[video, frame, view], row=row, gt=gt, identity=ref, native_action=action['action'],
                        taxonomy=tags, selected=rowinfo.get(row, {}).get('selected'),
                        correct_available=bool(support), identity_anchor_before=before,
                        system_already_contaminated_before=was_polluted, previously_seen_GT=seen,
                        gallery_before=action['gallery_before'], gallery_after=action['gallery_after'])
            sw = switch_keys.get((frame, view, row))
            if sw:
                if 'FALSE_SPLIT' in tags: reason = 'FALSE_SPLIT_NEW'
                elif 'FALSE_BIRTH' in tags: reason = 'FALSE_BIRTH_CORRECT_ACTIVE_MISSING_OR_UNKNOWN'
                elif 'WRONG_REACTIVATION' in tags: reason = 'WRONG_STALE_MERGE'
                elif 'FALSE_MERGE' in tags or rowinfo.get(row, {}).get('selected')=='Certified Wrong': reason = 'WRONG_EXISTING_MERGE'
                elif rowinfo.get(row, {}).get('selected')=='Selected UNKNOWN': reason = 'UNKNOWN_OR_CONTAMINATED_EXISTING_CHANGE'
                elif gt is None: reason = 'OFFLINE_CURRENT_GT_UNAVAILABLE'
                else: reason = 'PURE_FRAGMENT_SWITCH_OR_RECOVERY'
                count['IDSW/'+reason] += 1; info['CLEAR_IDSW'] = dict(sw, reason=reason)
            for tag in tags:
                count[tag] += 1
                if len(examples[tag])<4: examples[tag].append(info)
            if tags or sw: trace.append(info)
            observations[ref] += 1
            if gt is not None: votes[ref][gt] += 1; ever.add(gt)
    count['final_mixed_identities'] = sum(len(v)>1 for v in votes.values())
    count['native_committed_rows'] = sum(len(e['ids']) for e in commits)
    count['IDSW_exact'] = len(switches)
    assert sum(v for k, v in count.items() if k.startswith('IDSW/')) == len(switches)
    for x in trace:
        if x['gt'] is not None: episodes[x['gt'], x['key'][2]].append(x)
    spans = []
    for (gt, view), es in episodes.items():
        active = []
        for e in es:
            if active and e['key'][1] != active[-1]['key'][1]+1:
                spans.append(dict(gt=gt, view=view, start=active[0]['key'][1], end=active[-1]['key'][1], observed_error_payloads=len(active), right_censored=True)); active=[]
            active.append(e)
        if active: spans.append(dict(gt=gt, view=view, start=active[0]['key'][1], end=active[-1]['key'][1], observed_error_payloads=len(active), right_censored=True))
    path = (OUT/'forensics_v1' if artifact_root is None else Path(artifact_root))/out.parent.name/f'video{video:02d}'
    # Phase prefix distinguishes 20k and 24k even if case/seed names match.
    budget='24k' if 'onpolicy' in str(out) else '20k'
    if artifact_root is not None:budget='20k' if r['phase']=='formal' else '24k'
    path = path / budget
    path.mkdir(parents=True, exist_ok=True)
    with gzip.open(path/'EVENTS.jsonl.gz', 'wt') as f:
        for item in trace: f.write(json.dumps(item)+'\n')
    summary = dict(status='COMPLETE', variant=r['variant'], seed=r['seed'], phase=r['phase'], video=video,
        counts=dict(count), UNKNOWN_strata={k: dict(v) for k,v in strata.items()},
        logit_margins={k: quantiles(v) for k,v in margins.items()},
        exact_CLEAR_switch_reconstruction=switch_counts, consecutive_decision_diagnostics=dict(repeat),
        identity_error_intervals=spans, examples=dict(examples),
        event_stream={'path':str(path/'EVENTS.jsonl.gz'), 'SHA256':sha(path/'EVENTS.jsonl.gz')},
        evidence={'saved_result':str(out/'RESULT.json'), 'result_SHA256':sha(out/'RESULT.json'),
                  'raw_SHA256':sha(rawpath), 'commits_SHA256':r['commits_SHA256'], 'questions_SHA256':r['questions_SHA256'],
                  'checkpoint':r['trained']['checkpoint'] if r['trained'] else None,
                  'actor_source_commit':r['binding']['source_commit']},
        prefix_reproduction='rerun frozen actor source/checkpoint from bootstrap through key; full actor journals preserved, no claim an already-deleted mutable recovery slot is available',
        error_interval_scope='contiguous observed taxonomy events; gaps/end censored; false births, IDSW, merge events and propagation duration are distinct')
    save(path/'RESULT.json', summary)
    return summary

def main():
    protect(); begin=time.monotonic()
    ann=read('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json'); results=[]
    for phase, variant in [('formal','full'), ('formal','fixed_question'), ('formal','set_transformer'), ('formal','cosine'), ('onpolicy','full')]:
        seeds=[20261009] if variant=='cosine' else [20261008,20261009,20261010]
        for seed in seeds:
            for v in DEV:
                out=OLD/f'{phase}_online_v1'/f'{variant}_seed{seed}'/f'video{v:02d}'
                result=audit_case(out,ann,v); results.append(result)
                save(OUT/'forensics_v1/PROGRESS.json',dict(status='RUNNING', completed=len(results), total=39, phase=phase,variant=variant,seed=seed,video=v))
                print('FORENSIC_CASE_COMPLETE',phase,variant,seed,v,result['counts'],flush=True)
    grouped=collections.defaultdict(collections.Counter)
    for r in results:grouped[f"{r['phase']}/{r['variant']}"] .update(r['counts'])
    payload=dict(status='COMPLETE',binding=binding(),annotation_SHA256=sha('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json'),
                 cases=results, aggregate_counts={k:dict(v) for k,v in grouped.items()}, seconds=time.monotonic()-begin,
                 GT_scope='offline annotation only; natural separate mutated-state policies; descriptive attribution, not same-prefix causal interventions',
                 overlapping_taxonomy=True, IDSW_reconstruction='exact original TrackEval dataset preprocessing and CLEAR continuation-priority matching; every official IDSW assigned once')
    save(REPORTS/'ERROR_ATTRIBUTION.json',payload)
    save(REPORTS/'UNKNOWN_CHOICE_FORENSICS.json',dict(status='COMPLETE',binding=binding(),
        error_attribution_SHA256=sha(REPORTS/'ERROR_ATTRIBUTION.json'),
        training_deployment_semantics={'CE_Brier':'UNKNOWN masked from certified normalizer',
          'structured_loss':'legal & certified-known existing; UNKNOWN unavailable as loss-augmented competitor',
          'native_commit':'all lawful existing options remain eligible; no GT-based mask',
          'conclusion':'objective/actor support mismatch; attribution quantified, UNKNOWN never coerced negative'},
        cases=[{k:r[k] for k in ['variant','seed','phase','video','counts','UNKNOWN_strata','logit_margins','evidence']} for r in results]))
    save(REPORTS/'FALSE_MERGE_SPLIT_DIAGNOSTICS.json',dict(status='COMPLETE',binding=binding(),
        error_attribution_SHA256=sha(REPORTS/'ERROR_ATTRIBUTION.json'),aggregate_counts=payload['aggregate_counts'],
        definitions={'false_merge':'first introduction of another certified observed GT into a prior anchored native ID; mixed continuation separate',
         'false_split':'native new ID despite certified pure active target', 'false_birth':'native new ID for a GT seen earlier in any committed camera',
         'IDSW':'exact CLEAR change in last matched predicted ID; every change has one cause category',
         'unknown':'empty/mixed history never an automatically certified wrong identity'},
        formal_train_video16_question='development17/18/19 artifacts cannot establish TRAINvideo16 concentration; TRAIN own-state/dense diagnostics follow separately'))
    save(OUT/'forensics_v1/PROGRESS.json',dict(status='COMPLETE',completed=len(results),total=39))
    print('PHASE14_P1_COMPLETE',len(results),flush=True)

if __name__=='__main__':main()

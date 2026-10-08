"""GT-only post-inference diagnostics; never called by a native actor."""
import collections,gzip,json
import numpy as np
from jev_phase8_opportunity import PrefixIdentityAnchors
from gtr.modeling.jev_candidate_assignment import assign_candidate_values


def audit(aligned,journal,commits):
 anchors=PrefixIdentityAnchors();counts=collections.Counter();timelines=collections.defaultdict(dict);write_gts=collections.defaultdict(set);statuses={};records=[]
 with gzip.open(journal,'rt')as handle:packets={tuple(d['key'][1:]):d for d in map(json.loads,handle)}
 with gzip.open(commits,'rt')as handle:stream=list(map(json.loads,handle))
 for commit in stream:
  frame,view=commit['key'][1:];packet=packets.get((frame,view));mapping=anchors.reliable();fixed=None
  if packet:
   scores=np.asarray(packet['GMT_scores'],dtype=float);threshold=np.asarray(packet['thresholds'],dtype=float)
   fixed=assign_candidate_values(scores-threshold[None,:],np.zeros(len(scores)),packet['candidate_ids'],packet['legal_mask'])
  observed={};gtrows={}
  for row,t in enumerate(commit['ids']):
   key=(frame,view,row);r=aligned.get(key);gt=r['gt']if r else None;observed[row]=t;gtrows[row]=gt;event=commit['events'][row];status=None if gt is None or t not in mapping else mapping[t]==gt;statuses[key]=status
   counts['GT_unknown']+=gt is None;counts['anchor_unknown']+=gt is not None and t not in mapping
   counts['confirmed_correct_observations']+=status is True;counts['confirmed_wrong_observations']+=status is False
   if gt is not None:
    previous=timelines[(gt,view)].get(frame,False);bad=None if status is None else not status
    timelines[(gt,view)][frame]=True if previous is True or bad is True else None if previous is None or bad is None else False
   if event['birth']and gt is not None:counts['false_birth_observations']+=gt in anchors.last_GT_frame
   if event['write']and gt is not None:
    write_gts[t].add(gt);counts['wrong_gallery_writes']+=status is False;counts['unknown_gallery_writes']+=status is None
   if not packet:continue
   refs=packet['candidate_ids'];legal=np.asarray(packet['legal_mask'][row],dtype=bool);values=np.asarray(packet['candidate_values'][row]);support=[col for col,tref in enumerate(refs)if legal[col]and gt is not None and mapping.get(tref)==gt]
   counts['candidate_rows']+=1;counts['known_GT_candidate_rows']+=gt is not None
   if gt is not None:counts['candidate_miss_rows']+=not support;counts['available_correct_rows']+=bool(support)
   order=sorted(np.flatnonzero(legal),key=lambda col:(-values[col],refs[col]));chosen=int(order[0])if order else None
   counts['ranking_evaluable_rows']+=bool(support)
   if support:
    rank=min(order.index(col)+1 for col in support);counts['candidate_rank1_hits']+=rank==1;counts['candidate_MRR_sum']+=1/rank;counts['committed_correct_with_support']+=status is True
   initial=packet['initial_proposal_ids'][row];proposed=None if gt is None or initial not in mapping else mapping[initial]==gt;fixed_id=fixed.existing_ids[row];fc=None if gt is None or fixed_id not in mapping else mapping[fixed_id]==gt
   committed_existing=packet['existing_ids'][row];ec=None if gt is None or committed_existing not in mapping else mapping[committed_existing]==gt
   counts['selected_existing_changes_from_fixed']+=committed_existing!=fixed_id;counts['selected_existing_changes_from_GMT_proposal']+=committed_existing!=initial
   if fc is not None and ec is not None:
    counts['paired_existing_known_rows']+=1;counts['N01_same_prefix_fixed_to_policy']+=not fc and ec;counts['N10_same_prefix_fixed_to_policy']+=fc and not ec
   if proposed is not None and status is not None:
    counts['N01_GMT_proposal_to_native_commit']+=not proposed and status;counts['N10_GMT_proposal_to_native_commit']+=proposed and not status
   counts['unknown_existing_changed_rows']+=committed_existing!=fixed_id and(fc is None or ec is None)
   records.append({'key':[frame,view,row],'offline_GT':gt,'prefix_confirmed_commit_correct':status,'GMT_proposal_correct':proposed,'same_prefix_Fixed_existing_correct':fc,'policy_existing_correct':ec,'available_correct_candidates':len(support),'rank1_correct':chosen in support if support else None})
  anchors.update(observed,gtrows,frame)
 episodes=[]
 for (gt,view),line in sorted(timelines.items()):
  active=[];previous=None
  for frame,bad in sorted(line.items()):
   if active and(bad is not True or frame!=previous+1):episodes.append({'GT':gt,'view':view,'start':active[0],'end':active[-1],'duration_camera_frames':len(active),'right_censored':bad is None or frame!=previous+1});active=[]
   if bad is True:active.append(frame)
   previous=frame
  if active:episodes.append({'GT':gt,'view':view,'start':active[0],'end':active[-1],'duration_camera_frames':len(active),'right_censored':True})
 durations=[e['duration_camera_frames']for e in episodes];den=counts['ranking_evaluable_rows'];known=counts['known_GT_candidate_rows']
 return {'definition':'IoU>=.5 alignment after all inference; permanent first two consistent known prefix observations; same-camera payload labels evaluated before updating anchors. Unknown is neither correct nor wrong.','counts':dict(counts),'candidate_rank1':counts['candidate_rank1_hits']/den if den else None,'candidate_MRR':counts['candidate_MRR_sum']/den if den else None,'candidate_full_recall':counts['available_correct_rows']/known if known else None,'candidate_miss_rate':counts['candidate_miss_rows']/known if known else None,'committed_accuracy_with_correct_support':counts['committed_correct_with_support']/den if den else None,'wrong_ID_duration_total_camera_frames':sum(durations),'wrong_ID_duration_quantiles_camera_frames':dict(zip(['p50','p95','max'],map(float,np.quantile(durations,[.5,.95,1]))))if durations else {'p50':0.,'p95':0.,'max':0.},'contaminated_gallery_tracks':sum(len(gs)>1 for gs in write_gts.values()),'wrong_ID_episodes':episodes,'rows':records,'row_statuses':{','.join(map(str,k)):v for k,v in statuses.items()},'N01_N10_scope':'same live prefix, identical legal existing candidates, Fixed versus policy assignment; known existing identities only, unknown and NEW excluded. Descriptive current intervention, not future causal estimate.'}

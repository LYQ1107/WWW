"""Offline identity anchoring and global-feasibility diagnostics, no live actor."""
from collections import Counter,defaultdict
import numpy as np
from jev_phase8_common import ROOT
from gtr.modeling.jev_assignment import constrained_hungarian

class PrefixIdentityAnchors:
    def __init__(self):self.counts=defaultdict(Counter);self.first={};self.last_GT_frame={};self.confirmed={};self.confirmed_at={}
    def update(self,committed,gt_by_row,frame):
        for row,track in committed.items():
            gt=gt_by_row.get(int(row))
            if gt is None:continue
            track=int(track);gt=int(gt);self.first.setdefault(track,gt);self.counts[track][gt]+=1;self.last_GT_frame[gt]=int(frame)
            if sum(self.counts[track].values())==2 and len(self.counts[track])==1:
                self.confirmed[track]=self.first[track];self.confirmed_at[track]=int(frame)
    def reliable(self):
        return dict(self.confirmed)
    def strict_pure(self):
        return {t:self.first[t]for t,c in self.counts.items()if sum(c.values())>=2 and len(c)==1}
    def majority(self):return {t:c.most_common(1)[0][0]for t,c in self.counts.items()}
    def diagnostics(self):return {'counts':{str(t):dict(c)for t,c in self.counts.items()},'first':dict(self.first),'reliable':self.reliable(),'strict_pure':self.strict_pure(),'confirmed_at':dict(self.confirmed_at),'last_GT_frame':dict(self.last_GT_frame),'contract':'PERMANENT_FIRST_TWO_CONSISTENT_KNOWN_OBSERVATIONS'}

def forced_feasible_pairs(scores,banned,row,column):
    """Oracle-2 existence: fix one allowed edge, solve its residual exactly.

    This is offline feasibility, not a GT-aware deployed controller or proof
    that the current production triage hook can select the requested edge.
    """
    s=np.asarray(scores,dtype=np.float64);m,n=s.shape;banned=set(map(tuple,banned))
    if not(0<=row<m and 0<=column<n)or(row,column)in banned or not np.isfinite(s[row,column]):return None
    rr=[i for i in range(m)if i!=row];cc=[j for j in range(n)if j!=column]
    lookup_r={r:i for i,r in enumerate(rr)};lookup_c={c:j for j,c in enumerate(cc)}
    residual=s[np.ix_(rr,cc)];residual_banned=[(lookup_r[r],lookup_c[c])for r,c in banned if r in lookup_r and c in lookup_c]
    pairs={row:column};pairs.update((rr[r],cc[c])for r,c in constrained_hungarian(residual,residual_banned))
    assert len(set(pairs.values()))==len(pairs)
    assert all((r,c)not in banned and np.isfinite(s[r,c])for r,c in pairs.items())
    return pairs

def classify_row(track_ids,scores,pairs,row,gt,action,anchors,banned=(),mapping_override=None):
    mapping=anchors.reliable()if mapping_override is None else mapping_override;loose=anchors.majority();col=pairs.get(row);track=int(track_ids[col])if col is not None else None
    correct=[int(t)for t in track_ids if gt is not None and mapping.get(int(t))==gt]
    unanchored=[int(t)for t in track_ids if int(t)not in mapping]
    known_proposal=track in mapping if track is not None else False
    proposal_correct=(mapping.get(track)==gt)if gt is not None and known_proposal else None
    feasible={}
    for t in correct:
        c=list(track_ids).index(t);p=forced_feasible_pairs(scores,banned,row,c)
        if p is not None:feasible[t]=p
    strict_wrong=gt is not None and known_proposal and proposal_correct is False and action=='ACCEPT_CURRENT'
    loose_wrong=gt is not None and track in loose and loose.get(track)!=gt and action=='ACCEPT_CURRENT'
    loose_candidates=[int(t)for t in track_ids if gt is not None and loose.get(int(t))==gt]
    order=sorted(range(len(track_ids)),key=lambda c:(-float(scores[row][c]),c))
    ranks={int(track_ids[c]):rank+1 for rank,c in enumerate(order)}
    if gt is None:bucket='GT_UNKNOWN'
    elif action=='START_NEW':bucket='NEW_WITH_AVAILABLE_CORRECT_HISTORY'if correct else'NEW_WITHOUT_CONFIRMED_CORRECT_HISTORY'
    elif not known_proposal:bucket='PROPOSAL_IDENTITY_UNANCHORED_OR_MIXED'
    elif proposal_correct:bucket='PROPOSAL_CORRECT'
    elif not correct:bucket='WRONG_CANDIDATE_UNCERTAIN'if unanchored else'WRONG_CORRECT_CANDIDATE_ABSENT'
    elif not feasible:bucket='WRONG_CORRECT_CANDIDATE_GLOBAL_INFEASIBLE'
    else:bucket='WRONG_CORRECT_CANDIDATE_GLOBAL_FEASIBLE_NATIVE_NOT_YET_TESTED'
    return {'offline_GT':gt,'proposal_track_id':track,'proposal_correct':proposal_correct,'first_action':action,
       'correct_candidate_ids':correct,'correct_candidate_ranks':{t:ranks[t]for t in correct},'unanchored_or_mixed_candidate_ids':unanchored,
       'duplicate_GT_alias_ambiguity':len(correct)>1,'strict_wrong_with_feasible_correct_candidate':strict_wrong and bool(feasible),
       'loose_majority_wrong_with_candidate':loose_wrong and bool(loose_candidates),'bucket':bucket,'feasible_pairs':feasible,
       'row_candidate_order':order,'proposal_column':col,'candidate_count':len(track_ids),
       'top1_top2_margin':float(scores[row][order[0]]-scores[row][order[1]])if len(order)>1 else None,
       'confirmed_historical_IDs_with_mixed_history':[int(t)for t in track_ids if int(t)in mapping and len(anchors.counts[int(t)])>1]}

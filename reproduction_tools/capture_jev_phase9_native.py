"""Capture a sealed observational sample from live factual native GMT OFF."""
import argparse,copy,gzip,json,random,time
from collections import defaultdict,Counter
import numpy as np
import torch
from jev_phase9_common import *
from jev_phase8_opportunity import PrefixIdentityAnchors,classify_row
from jev_phase8_common import FrozenOFFActor
from jev_phase7_native import AttributionResolver
from jev_phase7_offline import IdentityEvaluator
from jev_phase7_state import CompleteStateDigest
from gtr.modeling.jev_state import count_track_history

def main(video):
 protect();protocol=json.loads((REPORTS/'PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json').read_text());source=binding();assert not source['dirty']
 expected_sha='5d8504af27a351346e683aeeec1bb67678a71249b213a181bd6bb020d788a966';assert sha(REPORTS/'PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json')==expected_sha
 random.seed(20261008);np.random.seed(20261008);torch.manual_seed(20261008);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 run=OUT/'capture_v1'/f'video{video:02d}'
 if (run/'CAPTURE_RESULT.json').exists():print('ALREADY_COMPLETE',video);return
 assert not run.exists(),'incomplete output must be retained; resume into versioned directory'
 run.mkdir(parents=True)
 selected=defaultdict(lambda:defaultdict(list))
 for d in protocol['selected_events']:
  if d['key'][0]==video:selected[tuple(d['key'])][d['row']].append(d)
 save(run/'START_MANIFEST.json',{'binding':source,'protocol_sha256':expected_sha,'events':sum(len(x)for x in selected.values()),'argv':sys.argv})
 lab=native_lab(video);evaluator=IdentityEvaluator(video);anchors=PrefixIdentityAnchors();digest=CompleteStateDigest();pending=None;snapshots=[]
 resolver=AttributionResolver(lab,FrozenOFFActor());lab.native_resolver=resolver;lab.engine.resolve_actions=resolver.resolve
 def choose(policy,feature,question,legal,off_action,context=None):
  lab.decision_rows.append({'question':question,'action':off_action,'original_action':off_action,'off_action':off_action,'legal':list(legal),'feature_vector':feature.detach().cpu().tolist(),'context':{k:v for k,v in(context or{}).items()if k!='tracker_state_before'},'probabilities':{a:float(a==off_action)for a in legal}});return off_action
 lab.pilot.choose=choose
 def gt_rows(payload):
  key=(video,int(payload['frame']),int(payload['view']));image=lab.pilot.image_for(lab.lookup,*key);assert (int(image['frame_id']),int(image['view_id']))==(key[1]+1,key[2]+1)
  preds=[];lab.original_append(preds,payload,image,{r:r+1 for r in range(len(payload['pred_boxes']))});aligned=evaluator.align(preds)
  return {r:aligned[(key[1],key[2],r)]['gt']for r in range(len(preds))}
 anchors.update(lab.seed_state.association_history[-1]['assignments'],gt_rows(lab.seed_payload),lab.seed_key[1])
 base=lab.original_propose
 def propose(payload,state,**kw):
  nonlocal pending
  proposal=base(payload,state,**kw)
  if state.reactivation_mode:return proposal
  key=(video,int(payload['frame']),int(payload['view']));gt=gt_rows(payload);pending={'key':key,'gt':gt,'snapshot':None}
  if key in selected:
   actions=lab.engine._actions_for_proposal(proposal,state);matrix=proposal.scores.detach().cpu().numpy()
   ds={r:classify_row(proposal.track_ids,matrix,proposal.pairs,r,gt.get(r),actions[r],anchors,proposal.banned_edges)for r in selected[key]}
   for r,d in ds.items():d['sampling_records']=selected[key][r]
   pending['snapshot']={'key':key,'state':state.clone(),'metadata':copy.deepcopy(lab.metadata),'complete_state_fingerprint':digest(state,lab.metadata),'proposal':copy.deepcopy(proposal),'offline_prefix_identity':anchors.diagnostics(),'descriptors':ds,'selected_rows':sorted(ds),'binding':source,'raw_score_sha256':__import__('hashlib').sha256(matrix.tobytes()).hexdigest()}
  return proposal
 lab.original_propose=propose
 def before(key,payload,state,kwargs,decisions):
  assert pending['key']==key
  if pending['snapshot']is None:return
  s=pending['snapshot'];s['decision_rows']=copy.deepcopy(decisions);s['factual_step_kwargs']=copy.deepcopy(kwargs);byrow={int(d['context']['detection_index']):d for d in decisions if d['question']=='MATCH_DECISION'};p=s['proposal'];s['online']={}
  for r,d in s['descriptors'].items():
   evidence=[];pc=p.pairs.get(r)
   for c,t in enumerate(p.track_ids):
    m=lab.metadata.get(t,{});emb=state.track_embeddings.get(t);obs=payload['reid_features'][r];cos=float(torch.nn.functional.cosine_similarity(obs.reshape(1,-1),emb.reshape(1,-1)))if emb is not None else 0.
    evidence.append([float(p.scores[r,c]),float(p.scores[r,c]-(p.scores[r,pc]if pc is not None else 0)),float(c==pc),float(sum(j==c for rr,j in p.pairs.items()if rr!=r)),float(count_track_history(state.association_history,t)),float(len(state.memory.get(t,[]))),float(max(0,key[1]-m.get('last_seen',key[1]))),len(m.get('views',[]))/2,float(t in state.active_ids),float(t in state.stale_ids),cos,float(emb is not None)])
   s['online'][r]={'state_features':byrow[r]['feature_vector'],'candidate_evidence':evidence,'legal_mask':[[rr,c]not in list(map(list,p.banned_edges))and bool(torch.isfinite(p.scores[rr,c]))for c in range(len(p.track_ids))for rr in [r]],'candidate_refs_metadata_only':list(p.track_ids)}
   mapping={int(t):int(g)for t,g in s['offline_prefix_identity']['reliable'].items()};g=d['offline_GT']
   d['candidate_correctness']=[None if g is None or int(t)not in mapping else bool(mapping[int(t)]==g)for t in p.track_ids]
   # NEW correctness stays unknown when any available history identity is unanchored.
   d['NEW_correctness']=None if g is None or any(int(t)not in mapping for t in p.track_ids)else not bool(d['correct_candidate_ids'])
   d['candidate_future_utility']=[None]*len(p.track_ids);d['NEW_future_utility']=None
  path=run/'snapshots'/f'STATE_{len(snapshots):03d}.pth';path.parent.mkdir(exist_ok=True);torch.save(s,path)
  snapshots.append({'key':list(key),'selected_rows':s['selected_rows'],'path':str(path),'sha256':sha(path),'sampling_records':{str(r):d['sampling_records']for r,d in s['descriptors'].items()}})
  save(run/'CAPTURE_PROGRESS.json',{'status':'RUNNING','video':video,'captured_states':len(snapshots),'captured_rows':sum(len(s['selected_rows'])for s in snapshots),'last_key':list(key)})
 def after(key,payload,state,result,kwargs,decisions):anchors.update(result['committed_track_ids'],pending['gt'],key[1])
 result=lab.run(run/'factual',before_step=before,after_step=after)
 old=json.loads((PREVIOUS/'opportunity_scan_v1'/f'video{video:02d}'/'SCAN_RESULT.json').read_text());actual=sha(result['predictions']);assert actual==old['factual_predictions_sha256'],'new factual capture changed original GMT OFF'
 observed={tuple(s['key'])+(r,)for s in snapshots for r in s['selected_rows']};wanted={k+(r,)for k,rows in selected.items()for r in rows};assert observed==wanted
 save(run/'CAPTURE_RESULT.json',{'status':'COMPLETE','binding':source,'video':video,'bounded_snapshots':snapshots,'max_frame':old['max_frame'],'original_phase8_GMT_OFF_predictions_exact_SHA_parity':True,'factual_predictions_sha256':actual,'original_scan_sha256':sha(PREVIOUS/'opportunity_scan_v1'/f'video{video:02d}'/'SCAN_RESULT.json'),'factual':result,'rows':len(observed),'no_new_future_outcomes_read':True})
 save(run/'CAPTURE_PROGRESS.json',{'status':'COMPLETE','video':video,'captured_states':len(snapshots),'captured_rows':len(observed)});print('CAPTURE_COMPLETE',video,len(observed),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);a=p.parse_args();main(a.video)

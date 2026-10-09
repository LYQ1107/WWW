"""Post-inference paired intervention, current Fixed then identical future actor."""
import collections,torch
from jev_phase12_common import *
from gtr.modeling.jev_candidate_policy import CandidateValuePolicy
from gtr.modeling.jev_native_state import NativeStateForkAdapter
from gtr.modeling.visual_jev_mcmot.baselines import FixedNativeController
from run_jev_phase10_closed_loop import raw_predictions

def policy(case):
 return None if case['kind']=='off' else CandidateValuePolicy('gmt_values')

def paired_h32(model,case,allinputs,images,evaluator,aligned,commits,prefixpath,out):
 from jev_phase8_opportunity import PrefixIdentityAnchors
 from jev_phase8_utility import effects
 anchors=PrefixIdentityAnchors();groups=collections.defaultdict(dict)
 for (frame,view,row),r in sorted(aligned.items()):
  if(frame,view)<(128,0):groups[frame,view][row]=r
 for (frame,view),rs in sorted(groups.items()):anchors.update({row:r['id']for row,r in rs.items()},{row:r['gt']for row,r in rs.items()},frame)
 expected={tuple(d['key'][1:]):d for d in commits};branches={};fixed=FixedNativeController();actual_controller=model.visual_jev_controller;model.jev_candidate_latency_observer=None
 for tag in ['actual','same_prefix_Fixed_current']:
  traces=[]
  def before(**d):
   intervene=tag!='actual' and (d['frame'],d['view'])==(128,0);model.visual_jev_enabled=intervene or case['kind']!='off';model.visual_jev_controller=fixed if intervene else actual_controller;model.jev_candidate_policy=CandidateValuePolicy('gmt_compat') if model.visual_jev_enabled else None;model.__dict__.pop('_jev_candidate_last',None)
  def after(**d):
   inst=d['instances'][-1];ids=inst.track_ids.detach().cpu().tolist();assert len(set(ids))==len(ids);record={'key':[int(d['frame']),int(d['view'])],'ids':ids,'id_count':int(d['id_count']),'hits':{str(t):int(d['hits'][t])for t in ids},'gallery_lengths':{str(t):len(d['galleries'][t])for t in ids}};traces.append(record)
   if tag=='actual':
    original=expected[tuple(record['key'])]
    assert all(record[k]==original[k]for k in ['ids','id_count','hits','gallery_lengths']),'paired actual H32 differs from complete online path'
  model.jev_native_prefix_observer=before;model.jev_candidate_commit_observer=after
  with torch.no_grad():raw,_=NativeStateForkAdapter(model).run(prefixpath,allinputs,stop_frame=159)
  preds=[p for p in raw_predictions(raw,images)if 128<=int(evaluator.images[p['image_id']]['frame_id'])-1<=159];save(out/f'PAIRED_H32_{tag}_PREDICTIONS.json',preds);observed=evaluator.align(preds)
  branches[tag]={'effects':{str(h):effects(observed,anchors.diagnostics(),128,h,set())for h in [8,16,32]},'native_commits':traces,'predictions_SHA256':sha(out/f'PAIRED_H32_{tag}_PREDICTIONS.json')}
 delta={str(h):{k:branches['actual']['effects'][str(h)][k]-branches['same_prefix_Fixed_current']['effects'][str(h)][k]for k in ['utility','utility_birth_zero','wrong_identity_duration_camera_frames']}for h in [8,16,32]}
 result={'status':'PASS','prefix_SHA256':sha(prefixpath),'fixed_boundary':[128,0],'branches':branches,'actual_minus_same_prefix_Fixed':delta,'H32_regret_against_executed_Fixed_alternative':max(0,-delta['32']['utility']),'actual_current_and_future_matches_complete_online_path':True,'same_future_policy':'actual frozen controller, freshly computed on each mutated branch','scope':'paired whole-current-payload legal assignment intervention at a predeclared GT-independent boundary; one boundary per video/controller. Regret only against this executed alternative, not an oracle over all candidates.','no_GT_actor_input':True};save(out/'PAIRED_H32_AUDIT.json',result);return {k:v for k,v in result.items()if k!='branches'}|{'branches_effects':{tag:b['effects']for tag,b in branches.items()}}


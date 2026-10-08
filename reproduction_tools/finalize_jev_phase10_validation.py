"""True TrackEval pooling and validation-only controls before heldout release."""
import collections,json
from jev_phase10_common import *
from run_jev_phase10_closed_loop import metrics


def collect(phase,cases,videos):
 results={};raw=[]
 for case in cases:
  name=case['name'];rs={};predictions=[]
  for video in videos:
   p=OUT/f'{phase}_closed_loop_v1'/name/f'video{video:02d}'/'RESULT.json';r=json.loads(p.read_text());assert r['status']=='COMPLETE'and r['case']==case and r['video']==video and not r['binding']['dirty'];assert sha(r['predictions']['path'])==r['predictions']['SHA256'];assert r['actual_mutated_state_online'];rs[str(video)]=r;raw.append({'path':str(p),'SHA256':sha(p),'source_commit':r['binding']['source_commit']});predictions.extend(json.loads(Path(r['predictions']['path']).read_text()))
  out=OUT/f'{phase}_pooled_v1'/name;out.mkdir(parents=True,exist_ok=True);p=out/'POOLED_PREDICTIONS.json';save(p,predictions);combined,evaluated=metrics(p,videos,out);results[name]={'case':case,'pooled_metrics':combined,'per_video':rs,'true_TrackEval_COMBINED_SEQ':True,'raw_pooled_metrics':{'path':str(evaluated/'metrics.json'),'SHA256':sha(evaluated/'metrics.json')}};save(out/'RESULT.json',results[name]);print('TRUE_POOL_COMPLETE',phase,name,combined,flush=True)
 return results,raw


def main():
 protect();assert json.loads((OUT/'VALIDATION_COMPLETE.json').read_text())['status']=='COMPLETE';p=REPORTS/'PHASE10_VALIDATION_CONTROLLER_PROTOCOL.json';protocol=json.loads(p.read_text());results,raw=collect('validation',protocol['cases'],VAL);selected=protocol['learned_selection'];dynamic=[c for c in protocol['cases']if c.get('policy')=='dynamic_fixed_new'];bestdynamic=min(dynamic,key=lambda c:(-results[c['name']]['pooled_metrics']['HOTA'],c['alpha']))
 candidates=[]
 for name in ['Fixed','Bidirectional',bestdynamic['name']]:candidates.append({'family':name,'cases':[name],'validation_pooled_HOTA':results[name]['pooled_metrics']['HOTA']})
 for architecture in ['CandidateMLP','CandidateDeepSets']:
  names=[c['name']for c in protocol['cases']if c.get('architecture')==architecture and c['supervision']==selected[architecture]['supervision']];assert len(names)==3;candidates.append({'family':architecture,'cases':names,'validation_pooled_HOTA':sum(results[n]['pooled_metrics']['HOTA']for n in names)/3})
 bestordinary=min(candidates,key=lambda c:(-c['validation_pooled_HOTA'],c['family']));reference=next(c for c in protocol['cases']if c['name']==bestordinary['cases'][0]);cases=[c for c in protocol['cases']if c['kind']!='rule'or c['name']in ['Fixed','Bidirectional',bestdynamic['name']]]
 report={'status':'COMPLETE','validation_videos':VAL,'cases':results,'raw_results':raw,'selected_dynamic_rule':bestdynamic,'ordinary_candidates':candidates,'strongest_ordinary_validation':bestordinary,'architecture_supervision_selection':selected,'selection_sources':'frozen candidate validation NLL for learned supervision; true pooled validation HOTA for rules and strongest ordinary family','heldout_sealed':True};save(REPORTS/'VALIDATION_CLOSED_LOOP_RESULTS.json',report)
 gatefiles=['NATIVE_STATE_PARITY.json','H32_CAUSAL_PARITY.json','DATA_ELIGIBILITY.json','TINY_OVERFIT.json','THREE_MODEL_TRAINING_COMPLETE.json'];gates={}
 for name in gatefiles:
  d=json.loads((REPORTS/name).read_text());assert d['status']in ['PASS','COMPLETE'];gates[name]=sha(REPORTS/name)
 checkpoints=[c['checkpoint']for c in cases if c['kind']=='model']
 for checkpoint in checkpoints:assert sha(checkpoint['path'])==checkpoint['SHA256']
 freeze={'status':'FROZEN','heldout_videos':HELDOUT,'validation_videos':VAL,'cases':cases,'gate_SHA256':gates,'checkpoints':checkpoints,'validation_protocol_SHA256':sha(p),'validation_results_SHA256':sha(REPORTS/'VALIDATION_CLOSED_LOOP_RESULTS.json'),'strongest_ordinary_validation':bestordinary,'selected_dynamic_rule':bestdynamic,'learned_selection':selected,'latency_reference':reference,'latency_reference_seed_rule':'first preregistered seed20261008 if best ordinary is learned; no heldout timing selection','all_supervisions_all_seeds_main_architectures':True,'structural_ablations_supervision':selected['CandidateJEV']['supervision'],'candidate_TopK':'Full','NEW':'same fixed native confidence anchor and legal private dummy','architecture_gate':{'HOTA_points':.1,'AssA_points':.2,'positive_videos':2,'net_corrections_nonnegative':True,'p95_extra_ms':10,'extra_VRAM_MiB':128},'heldout_not_used_for_any_choice':True,'official_TEST':False,'Full24':False};save(REPORTS/'PHASE10_CHECKPOINT_AND_HELDOUT_FREEZE.json',freeze);print('CHECKPOINT_CONTROLS_FROZEN_HELDOUT_RELEASE_READY',len(cases),flush=True)
if __name__=='__main__':main()

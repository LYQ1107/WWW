"""Isolated real native-prefix feature/reader/assignment latency after all actors."""
import argparse,time,json,os
import numpy as np
import torch
from jev_phase12_common import *
from run_jev_phase12_closed_loop import configure

def main(gpu,repeats,fast=False):
    protect();source=binding();assert not source['dirty'];torch.set_num_threads(1)
    protocol=json.loads((REPORTS/'ONLINE_PROTOCOL.json').read_text());samples=[];checks=[]
    selections=[]
    for video in VAL:
        entries=json.loads((PREVIOUS/'native_capture_v1'/f'video{video:02d}/compat/RESULT.json').read_text())['prefixes'];entries=sorted(entries,key=lambda e:e['key'])
        selections.extend(entries[i] for i in [0,len(entries)//2,len(entries)-1])
    assert len(selections)==9
    model=build_model(17);folder='isolated_latency_fast_v1' if fast else 'isolated_latency_v1'
    for case in [c for c in protocol['cases'] if c['name']=='Fixed' or (c['kind']=='model' and c['model'] in ['full','set_transformer','question_plain','visual_deepsets','numerical_only','CandidateMLP'] and c.get('risk',True))]:
        configure(model,case);controller=model.visual_jev_controller
        if fast and case['kind']=='model':
            from gtr.modeling.visual_jev_mcmot.fast_identity_history_tokens import build_match_inputs,native_time_metadata
            controller.token_builder=build_match_inputs;controller.metadata_builder=native_time_metadata
        for entry in selections:
            key=entry['key'];packet=torch.load(entry['packet_path'],map_location='cuda:0');prefix=torch.load(entry['path'],map_location='cuda:0');batch=packet['packet']['batch'];frame,view=key[1:];observations=prefix['instances'][frame*2+view].reid_features
            controller.native_context(prefix['instances'],prefix['galleries'],frame,view,prefix['first'],video_id=key[0])
            for iteration in range(repeats+3):
                torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
                with torch.no_grad():controller.match(batch,None,prefix['galleries'],observations)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)*1000;peak=torch.cuda.max_memory_allocated();stage=controller.timings[-1]
                if iteration>=3:samples.append({'case':case['name'],'key':key,'questions':len(observations),'candidates':len(batch.candidate_ids),'total_policy_and_feature_ms':elapsed,**stage,'temporary_peak_MiB':(peak-base)/1048576,'absolute_peak_MiB':peak/1048576})
                controller.records.clear();controller.timings.clear()
            checks.append({'case':case['name'],'key':key,'native_prefix_SHA256':entry['sha256'],'checkpoint_SHA256':case.get('checkpoint',{}).get('SHA256'),'no_backbone_recompute':True})
            save(OUT/folder/'PROGRESS.json',{'status':'RUNNING','case':case['name'],'last_key':key,'samples':len(samples),'fast_tokens':fast})
    summaries={}
    for name in sorted({r['case'] for r in samples}):
        subset=[r for r in samples if r['case']==name];summaries[name]={field:dict(zip(['p50','p95','max'],map(float,np.quantile([r[field] for r in subset],[.5,.95,1])))) for field in ['total_policy_and_feature_ms','feature_ms','policy_assignment_ms','temporary_peak_MiB','absolute_peak_MiB']}
    result={'status':'COMPLETE','binding':source,'gpu':gpu,'repeats_per_prefix':repeats,'selections':'first/median/last registered prefix per pre-frozen validation video; no metric-dependent selection','summaries':summaries,'provenance':checks,'scope':'real frozen native prefix inputs; isolated benchmark, not a new MOT run; total includes token building and Python lawful assignment; stage CUDA synchronization only at full-call boundaries','cache':'shared state encoded once per actual current-camera question batch, no cross-stage reuse after state commits','GPU_occupancy_check':'caller must ensure other tasks absent; process inventory recorded externally','samples':samples}
    result['fast_tokens']=fast;result['main_online_architecture_or_action_rules_changed']=False
    save(OUT/folder/'RESULT.json',result);print('REAL_NATIVE_PREFIX_LATENCY_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--repeats',type=int,default=10);p.add_argument('--fast',action='store_true');a=p.parse_args();main(a.gpu,a.repeats,a.fast)

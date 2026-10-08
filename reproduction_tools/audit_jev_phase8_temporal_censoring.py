"""Recompute censor masks from immutable native IDs; never rerun interventions."""
import gzip,json
import torch
from jev_phase8_common import *
from jev_phase8_utility import effects

def main():
    protect();start=binding();assert not start['worktree_dirty']
    audit=json.loads((REPORTS/'NATIVE_CORRECTIVE_ORACLE_AUDIT.json').read_text());out=OUT/'temporal_censor_v2';assert not out.exists();out.mkdir()
    changed=0;branches=0;gt={};prefixes={};origin={};records=[]
    for video in(12,13,14,16,17,18,19):
        directory=OUT/'opportunity_scan_v1'/f'video{video:02d}'
        with gzip.open(directory/'natural_event_index.jsonl.gz','rt')as f:
            gt[video]={tuple(e['key'][1:])+(e['row'],):e['offline_GT']for e in map(json.loads,f)}
        manifest=json.loads((directory/'SCAN_RESULT.json').read_text())
        for s in manifest['bounded_snapshots']:origin[tuple(s['key'])]=s
    for e in audit['events']:
        key=tuple(e['key']);video=key[0];s=origin[key]
        if key not in prefixes:
            snapshot=torch.load(s['path'],map_location='cpu');prefixes[key]=snapshot['offline_prefix_identity']
        for tag,b in e['branches'].items():
            candidates=[a for a in audit['raw_manifest']if a['sha256']==b['complete_state_trace_sha256']and a['path'].endswith('/COMPLETE_STATE_TRACE.json')]
            assert candidates;path=Path(candidates[0]['path']);trace=json.loads(path.read_text())
            rows={(r['key'][1],r['key'][2],int(i)):{'id':int(t),'gt':gt[video][(r['key'][1],r['key'][2],int(i))]}for r in trace for i,t in r['ids'].items()}
            original=json.loads((path.parent/'EFFECTS.json').read_text())['horizons'];new={}
            for h in(8,16,32):
                value=effects(rows,prefixes[key],key[1],h,{e['offline_GT_metadata']});old=original[str(h)]
                for field in('counts','target_counts','utility','utility_birth_zero','sensitivity','wrong_identity_duration_camera_frames'):assert value[field]==old[field],('non-censor metric changed',key,tag,h,field)
                assert len(value['wrong_identity_episodes'])==len(old['wrong_identity_episodes'])
                changed+=sum(a['right_censored']!=b['right_censored']for a,b in zip(value['wrong_identity_episodes'],old['wrong_identity_episodes']))
                new[str(h)]=value
            branches+=1;dest=out/f'video{video:02d}_{key[1]:05d}_{key[2]}_row{e["row"]:03d}_{tag}.json';save(dest,new)
            records.append({'key':list(key),'row':e['row'],'branch':tag,'original_effects_sha256':sha(path.parent/'EFFECTS.json'),'corrected_temporal_effects_path':str(dest),'corrected_sha256':sha(dest)})
    save(REPORTS/'TEMPORAL_CENSORING_AUDIT.json',{'status':'COMPLETE_METRIC_CENSOR_CORRECTION','binding':start,'events':len(audit['events']),'branches':branches,
        'changed_episode_endpoint_censor_flags_across_horizons':changed,'all_primary_and_sensitivity_utilities_unchanged':True,'all_wrong_known_duration_counts_unchanged':True,
        'native_candidate_gate_counts_unchanged':True,'native_interventions_rerun':False,'old_raw_effects_preserved':True,'raw_sidecars_uploaded':False,'raw_sidecar_manifest':records,
        'definition':'GT/identity-unknown observations censor known-wrong episodes; they cannot be counted as proven recovery. Observed anchored wrong camera-frame totals are lower bounds when identity/GT is unknown, not complete lifetime durations.'})
    protect();print(json.dumps({'status':'PASS','branches':branches,'censor_flags_corrected':changed,'utility_and_gate_unchanged':True}))
if __name__=='__main__':main()

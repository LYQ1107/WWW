"""Recover completed forensic logs after an original-reference path error."""
import argparse
import collections
import gzip
from jev_phase16_common import *
import torch
from jev_phase13_runtime import cache_inputs

def main(video):
    protect();assert video in TRAIN
    folder=OUT/'P0_r2/full/original'/f'video{video:02d}'
    assert not (folder/'RESULT.json').exists()
    log=OUT/'P0_queue_r2'/f'full_original_video{video}.log'
    expected_missing=str(XV/'commitment_dataset_v2'/f'video{video:02d}/RAW_PREDICTIONS.json')
    text=log.read_text()
    assert 'FileNotFoundError' in text and expected_missing in text,'only the exact post-inference reference error is recoverable'
    queries=folder/'FULL_QUERIES.jsonl.gz'
    values,frames,reader=cache_inputs(video);labels=duplicate_masked_labels(video,reader)
    sizes={view:len(reader.load(video,0,view)['pred_boxes']) for view in [0,1]}
    bootstrap=max((n,v) for v,n in sizes.items())[1]
    ids={(0,bootstrap):list(range(1,sizes[bootstrap]+1))}
    votes=collections.defaultdict(collections.Counter);observations=collections.Counter()
    for identity,gt in zip(ids[0,bootstrap],labels.current(0,bootstrap)):
        observations[identity]+=1
        if gt is not None:votes[identity][gt]+=1
    pending={};counts=collections.Counter();flags=collections.Counter();primary=collections.Counter()
    tasks=collections.defaultdict(collections.Counter);clusters=collections.defaultdict(set)
    with gzip.open(queries,'rt') as stream:
        for line in stream:
            r=json.loads(line);key=tuple(r['key'][1:]);row=r['row']
            if r.get('record_type')=='COMMIT':
                query=pending.pop((key,row));identity=r['committed_ID'];gt=query['current_GT_OFFLINE_ONLY']
                target=ids.setdefault(key,[])
                assert row==len(target),'native rows must remain ordered'
                target.append(identity);observations[identity]+=1
                if gt is not None:votes[identity][gt]+=1
                counts[r['action']]+=1
                continue
            pending[key,row]=r;gt=r['current_GT_OFFLINE_ONLY'];chosen=r['selected_ID']
            known=chosen is not None and sum(votes[chosen].values())>=3 and sum(votes[chosen].values())/max(1,observations[chosen])>=.8 and len(votes[chosen])==1
            counts['all_queries']+=1;counts['high_risk_queries']+=r['high_risk']
            counts['high_risk_GT_known_queries']+=r['high_risk'] and gt is not None
            counts['current_GT_known']+=gt is not None
            counts['observed_correct_content_exists']+=bool(r['raw_target_content_IDs'])
            counts['qualified_raw_content_exists']+=bool(r['qualified_raw_target_IDs'])
            counts['anchored_correct_owner_exists']+=bool(r['anchored_correct_owner_IDs'])
            counts['legal_pure_correct_available']+=bool(r['legal_pure_correct_IDs'])
            counts['selected_certified_correct']+=r['selected_certified_correct']
            counts['selected_certified_wrong']+=known and not r['selected_certified_correct'] and gt is not None
            counts['selected_UNKNOWN_history']+=chosen is not None and not known and gt is not None
            counts['selected_globally_mixed']+=chosen is not None and len(votes[chosen])>1 and gt is not None
            counts['summary_hidden_queries']+=bool(r['summary_hidden_IDs'])
            counts['owner_anchored_summary_hidden_queries']+=bool(r['owner_anchored_summary_hidden_IDs'])
            counts['raw_known_observations_at_query']+=sum(sum(v.values()) for v in votes.values())
            counts['raw_UNKNOWN_observations_at_query']+=sum(observations.values())-sum(sum(v.values()) for v in votes.values())
            tasks[r['task']]['queries']+=1;tasks[r['task']]['high_risk']+=r['high_risk']
            if r['high_risk']:
                primary[r['primary']]+=1
                for name,present in r['flags'].items():flags[name]+=present
            cluster=(video,gt,key[1],key[0]//64)
            if gt is not None and r['owner_anchored_summary_hidden_IDs']:
                clusters['owner_anchored_summary_loss'].add(cluster)
                if r['high_risk']:clusters['high_risk_owner_anchored_summary_loss'].add(cluster)
            if r['high_risk'] and gt is not None:clusters['high_risk_known'].add(cluster)
    assert len(ids)==2*frames and all(len(v)==len(set(v)) for v in ids.values())
    predictions=[]
    for key,tracks in sorted(ids.items()):
        frame,view=key;payload=reader.load(video,frame,view);image=labels.images[key]
        assert len(tracks)==len(payload['pred_boxes'])
        sx=image['width']/payload['image_size'][1];sy=image['height']/payload['image_size'][0]
        for box,identity,score in zip(payload['pred_boxes'].tolist(),tracks,payload['detection_scores'].tolist()):
            x,y,x2,y2=box;predictions.append(dict(image_id=image['id'],track_id=identity,
                bbox=[x*sx,y*sy,max(0.,(x2-x)*sx),max(0.,(y2-y)*sy)],score=float(score),category_id=1))
    baseline=read(XV/'train_commitment_prefixes_v1'/f'video{video:02d}/RESULT.json')['raw_predictions']
    assert sha(baseline['path'])==baseline['SHA256']
    assert predictions==read(baseline['path']),'complete reconstructed native predictions differ from original'
    counts['payloads']=2*frames-1
    summary=dict(counts=dict(counts),task_counts={k:dict(v) for k,v in tasks.items()},
        primary_high_risk=dict(primary),multi_label_high_risk=dict(flags),
        cluster_counts={k:len(v) for k,v in clusters.items()},
        clusters={k:[list(x) for x in sorted(v)] for k,v in clusters.items()},
        overlap_is_not_independent=True,GT_used_only_after_native_scores=True,
        raw_target_content_does_not_certify_Global_ID=True)
    actor=read(OUT/'source_P0_r2.json')
    checkpoint=read(XVROOT/'reports/JEV_PHASE15/SOURCE_AND_CHECKPOINT_MANIFEST.json')
    from jev_phase15_common import PHASE14
    training=PHASE14/'formal_v2/multi_question/availability_joint/seed20261009/RESULT.json'
    ck=read(training)['checkpoint']
    source=binding(checkpoints=[ck],inputs=[ref(log),ref(queries),baseline,ref(OUT/'source_P0_r2.json')],
        native_state=dict(kind='actual observed all-query native Gallery/Bank/commit trace',trace=ref(queries),
            complete_restorable_prefix_captured=False),
        evaluator='immutable native commits plus exact frozen detector geometry/score reconstruction',
        scope='post-inference aggregation reference repair, zero new tracker/GPU/optimizer steps')
    source['actual_actor']=actor;source['actual_recovery']=dict(react_learned=False,cosine_slots=[0,1,2],DEFER_logit=.75)
    case=dict(tag='FULL',summary=summary,queries=ref(queries),parity='FULL_RAW_PREDICTIONS_EXACT',
        native_start='unchanged native scene bootstrap; full restorable startup snapshot was not captured in this audit',
        native_start_SHA256=None,strict_train_causal_gate_eligible=video!=14,
        final_observed_identity_memory_SHA256=None,final_memory_hash_unavailable_after_aggregation_exception=True)
    result=dict(status='COMPLETE',binding=source,video=video,policy='original',scope='full',frames=frames,cases=[case],
        all_queries_not_selected_examples=True,seconds=None,any_new_training_updates=False,
        completed_native_trace_recovered_after_reference_path_error=True,original_failed_log=ref(log),
        exact_original_predictions=baseline,not_a_second_native_experiment=True)
    save(folder/'RESULT.json',result);save(folder/'PROGRESS.json',dict(status='COMPLETE_POSTPROCESSING_REPAIR',
        recovered_result=ref(folder/'RESULT.json'),no_native_rerun=True))
    print('PHASE16_COMPLETED_ORIGINAL_AUDIT_RECOVERED',video,len(predictions),dict(counts),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);main(p.parse_args().video)

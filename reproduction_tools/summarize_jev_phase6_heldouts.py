"""Report every preregistered heldout and pooled TrackEval, including native correction."""
import json
from pathlib import Path
from jev_phase6_common import OUT,REPORTS,ROOT,sha,save,new_output,protect_anchor
import run_early_pilot_tracking as pilot

def main():
    videos=[2,3,5];conditions=['G0','G1','G2','G3','G5','G5_NATIVE_TRANSITION']
    protocol=json.loads((REPORTS/'HELDOUT_PROTOCOL.json').read_text())
    if protocol['video_ids']!=videos:raise AssertionError('heldout selection changed')
    annotation_path=Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json')
    annotation=json.loads(annotation_path.read_text())
    images=[i for i in annotation['images'] if int(i['video_id']) in videos];image_ids={int(i['id']) for i in images}
    subset={**annotation,'images':images,'annotations':[a for a in annotation['annotations'] if int(a['image_id']) in image_ids]}
    results={};pooled={};baselines={}
    for condition in conditions:
        rows={};predictions=[]
        for video in videos:
            source=OUT/f'native_b2/video{video:02d}/result.json' if condition=='G5_NATIVE_TRANSITION' else OUT/f'heldouts/video{video:02d}/{condition}/result.json'
            record=json.loads(source.read_text());metrics=record['metrics']
            prediction=Path(record['result']['predictions'])
            actual=json.loads(prediction.read_text());predictions.extend(actual)
            if not {int(p['image_id']) for p in actual}.issubset(image_ids):raise AssertionError('heldout prediction crosses declared TRAIN sequences')
            rows[str(video)]={'metrics':metrics,'prediction_sha256':sha(prediction),'prediction_rows':len(actual),
                'source_result_sha256':sha(source),'checkpoint_sha256':record.get('checkpoint_sha256'),
                'source':str(source),'complete_sequence':True}
            if condition=='G0':baselines[video]=metrics
            delta={key:metrics[key]-baselines[video][key] for key in ['HOTA','AssA','IDF1','MOTA','IDSW']}
            rows[str(video)].update(delta_vs_GMT=delta,positive_both_primary=delta['HOTA']>0 and delta['AssA']>0)
        output=OUT/f'heldout_pooled/{condition}'
        if not (output/'result.json').exists():
            new_output(output);pilot.PILOT=output
            prediction=output/'pooled_predictions.json';prediction.write_text(json.dumps(predictions))
            dataset=pilot.prepare_eval_dataset(subset);_,evaluated=pilot.run_eval('pooled',prediction,dataset)
            save(output/'result.json',{'status':'COMPLETE','condition':condition,'video_ids':videos,
                'metrics':pilot.extract_metrics(evaluated),'prediction_sha256':sha(prediction),
                'aggregation':'TrackEval combined-sequence metrics, not arithmetic mean of HOTA','official_test_read':False})
        pooled[condition]=json.loads((output/'result.json').read_text())
        results[condition]={'per_video':rows,'positive_sequences':sum(r['positive_both_primary'] for r in rows.values()),
            'all_three_positive':all(r['positive_both_primary'] for r in rows.values())}
    for condition in conditions:
        delta={k:pooled[condition]['metrics'][k]-pooled['G0']['metrics'][k] for k in ['HOTA','AssA','IDF1','MOTA','IDSW']}
        results[condition].update(pooled_metrics=pooled[condition]['metrics'],pooled_delta_vs_GMT=delta,
            preregistered_gate_pass=condition!='G0' and results[condition]['all_three_positive'] and delta['HOTA']>0 and delta['AssA']>0)
    report={'status':'COMPLETE','primary_preregistered_runs':15,'native_supplement_runs':3,'video_ids':videos,
        'conditions':results,'heldout_protocol_sha256':sha(REPORTS/'HELDOUT_PROTOCOL.json'),
        'native_supplement_protocol_sha256':sha(REPORTS/'NATIVE_SUPPLEMENT_PROTOCOL.json'),
        'same_B2_checkpoint_for_legacy_and_native':True,'native_supplement_reason':'fix documented native learned second-round commit mismatch; correction protocol recorded before new outcomes',
        'model_or_hyperparameters_selected_from_outcomes':False,'video01_is_independent_test':False,
        'claim_scope':protocol['claim_scope'],'official_test_read':False,'full24_authorized':False,
        'what_did_we_learn':'Native state-transition fidelity changes heldout conclusions: native B2 improves videos02 and05 but slightly loses03; binary/scalar controls also fail the all-three rule. Preserve every primary and corrected result; no complete generalization or three-action necessity claim follows.'}
    save(REPORTS/'HELDOUT_RESULTS.json',report);protect_anchor();print(json.dumps({c:{'positive_sequences':r['positive_sequences'],'gate':r['preregistered_gate_pass'],'pooled':r['pooled_metrics']} for c,r in results.items()}))

if __name__=='__main__':main()

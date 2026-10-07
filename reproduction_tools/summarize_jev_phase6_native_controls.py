"""Pool fixed-weight mechanism controls without revising the original gates."""
import json
from pathlib import Path
from jev_phase6_common import OUT,REPORTS,B2,sha,save,new_output,protect_anchor
import run_early_pilot_tracking as pilot

def main():
    videos=[2,3,5];conditions=['G4a','G4b','VALIDATION_ONLY']
    annotation=json.loads(Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json').read_text())
    images=[i for i in annotation['images'] if int(i['video_id']) in videos];ids={int(i['id']) for i in images}
    subset={**annotation,'images':images,'annotations':[a for a in annotation['annotations'] if int(a['image_id']) in ids]}
    native=json.loads((REPORTS/'HELDOUT_RESULTS.json').read_text())['conditions']['G5_NATIVE_TRANSITION']
    results={}
    for condition in conditions:
        per_video={};predictions=[]
        for video in videos:
            path=OUT/f'native_removal/video{video:02d}/{condition}/result.json';record=json.loads(path.read_text())
            if record['checkpoint_sha256']!=sha(B2):raise AssertionError('control weights differ')
            if condition=='VALIDATION_ONLY':
                record['future']='live frozen B2; original assignment retained for native second validation'
                save(path,record)
            prediction=Path(record['result']['predictions']);rows=json.loads(prediction.read_text());predictions.extend(rows)
            if not {p['image_id'] for p in rows}.issubset(ids):raise AssertionError('non-preregistered sequence')
            baseline=native['per_video'][str(video)]['metrics']
            per_video[str(video)]={'metrics':record['metrics'],'prediction_sha256':sha(prediction),'source_result_sha256':sha(path),
                'native_B2_advantage':{k:baseline[k]-record['metrics'][k] for k in ['HOTA','AssA','IDF1','MOTA','IDSW']}}
        output=OUT/f'native_control_pooled/{condition}'
        if not (output/'result.json').exists():
            new_output(output);pilot.PILOT=output
            prediction=output/'pooled_predictions.json';prediction.write_text(json.dumps(predictions))
            dataset=pilot.prepare_eval_dataset(subset);_,evaluation=pilot.run_eval('pooled',prediction,dataset)
            save(output/'result.json',{'status':'COMPLETE','condition':condition,'metrics':pilot.extract_metrics(evaluation),
                'prediction_sha256':sha(prediction),'video_ids':videos,'aggregation':'actual combined-sequence TrackEval'})
        pooled=json.loads((output/'result.json').read_text())['metrics'];base=native['pooled_metrics']
        results[condition]={'per_video':per_video,'pooled_metrics':pooled,
            'native_B2_pooled_advantage':{k:base[k]-pooled[k] for k in ['HOTA','AssA','IDF1','MOTA','IDSW']}}
    report={'status':'COMPLETE','fixed_weight_control_runs':9,'videos':videos,'conditions':results,
        'frozen_B2_sha256':sha(B2),'original_development_gate_revised':False,'original_three_sequence_gate_revised':False,
        'GLOBAL_REASSOCIATION_EVIDENCE':'POSITIVE_ON_VIDEO02_CONTEXT_DEPENDENT',
        'native_second_validation_alone_explains_video02_gain':False,
        'attribution_limit':'Removing REASSOCIATE removes both global re-solve and binary revalidation. VALIDATION_ONLY preserves the latter, isolating the reassignment contribution. Subsequent closed-loop states recompute live.',
        'novelty_limit':'Frozen global reassignment helps a heldout sequence materially; this does not establish universal superiority over binary JEV, all-three improvement, useful learned MEMORY, or Full Lifecycle superiority.',
        'official_test_read':False,'full24_authorized':False,
        'what_did_we_learn':'Native global re-solving matters on video02: preserving second validation while removing global reassignment loses about1.29 HOTA/2.47 AssA. Contributions are tiny on03 and slightly negative on05. A blanket all-gain-is-gating conclusion is too strong, while the original development and full-lifecycle gates still fail.'}
    save(REPORTS/'NATIVE_HELDOUT_MECHANISM_AUDIT.json',report);protect_anchor()
    print(json.dumps({c:r['native_B2_pooled_advantage'] for c,r in results.items()}))

if __name__=='__main__':main()

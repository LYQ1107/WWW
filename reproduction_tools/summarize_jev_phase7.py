"""True pooled TrackEval plus offline identity/attribution diagnostics."""
import argparse,json,os
from jev_phase7_common import *

def summarize(tags):
    from jev_phase7_offline import IdentityEvaluator,mechanism_correctness
    ann=json.loads(ANNOTATIONS.read_text());selected={9,10,11}
    images=[i for i in ann['images']if int(i['video_id'])in selected];image_ids={i['id']for i in images}
    subset={'images':images,'annotations':[a for a in ann['annotations']if a['image_id']in image_ids],
            'categories':ann['categories'],'videos':ann['videos']}
    output={}
    for tag in tags:
        prior=REPORTS/'individual_results'/f'{tag}.json'
        if prior.exists():output[tag]=json.loads(prior.read_text());continue
        runs=[];predictions=[];diagnostics={}
        for video in (9,10,11):
            root=OUT/'closed_loop'/f'video{video:02d}'/tag
            byte=tag in ('BYTETRACK','BYTETRACK_PAPER06')
            if byte:root=OUT/('bytetrack'if tag=='BYTETRACK'else 'bytetrack_paper06')/f'video{video:02d}'
            result=json.loads((root/'result.json').read_text())
            assert result.get('complete_video',byte)
            path=result['predictions']if byte else result['result']['predictions']
            p=json.loads(Path(path).read_text());predictions.extend(p);runs.append(result)
            evaluator=IdentityEvaluator(video);rows=evaluator.align(p)
            audit=evaluator.summarize(rows)
            if 'mechanisms'in result:audit['mechanism_correctness']=mechanism_correctness(result['mechanisms'],rows)
            diagnostics[str(video)]=audit
        dest=OUT/'pooled'/tag
        if dest.exists():raise RuntimeError('refusing reused pooled output without finished summary')
        dest.mkdir(parents=True);path=dest/'predictions.json';save(path,predictions)
        os.environ['JEV_PILOT_ROOT']=str(dest)
        import run_early_pilot_tracking as pilot
        pilot.PILOT=dest
        dataset=pilot.prepare_eval_dataset(subset);prepared,evaluated=pilot.run_eval('jev',path,dataset)
        pooled=pilot.extract_metrics(evaluated)
        value={'tag':tag,'status':'COMPLETE','runs':runs,'true_pooled_TrackEval':pooled,
               'offline_identity_diagnostics':diagnostics,'pooled_predictions':str(path),
               'pooled_prediction_sha256':sha(path),'prepared':str(prepared),'evaluation':str(evaluated),
               'binding':binding(),'pooled_definition':'one evaluator over all six camera sequences; not a mean of video metrics',
               'official_test_read':False}
        save(prior,value);save(dest/'result.json',value);output[tag]=value
    save(REPORTS/'HELDOUT_RESULTS.json',{'status':'PARTIAL'if len(tags)<7 else 'PRIMARY_WITH_REQUESTED_DIAGNOSTICS',
                                     'results':output,'controller_heldouts':[9,10,11],'official_test_read':False})
    print(json.dumps({tag:v['true_pooled_TrackEval']for tag,v in output.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tags',nargs='+');summarize(p.parse_args().tags)

"""Offline coordinate and full-future emission-filter regression tests."""
import json
from pathlib import Path
import tempfile
import torch
from build_jev_counterfactual_v2 import build_v2_records,detection_target


def main():
    images={(1,1,1):10,(1,1,2):11}
    meta={10:{'width':32,'height':32},11:{'width':32,'height':32}}
    gt={10:[(7,[0,0,10,10])],11:[(8,[20,20,30,30])]}
    kwargs={'video_id':1,'view':0,'image_size':(32,32),'gt_by_image':gt,'image_meta':meta}
    aligned={(v,w,f-1):i for (v,w,f),i in images.items()}
    assert detection_target(frame=0,box=[0,0,10,10],images=images,**kwargs) is None
    assert detection_target(frame=0,box=[0,0,10,10],images=aligned,**kwargs)==7
    assert detection_target(frame=1,box=[20,20,30,30],images=aligned,**kwargs)==8
    class Cache:
        def keys(self):return [(1,f,0) for f in range(3)]
        def load(self,v,f,w):return {'video_id':v,'frame':f,'view':w,'cache_version':'fixture','pred_boxes':torch.tensor([[0.,0.,10.,10.]]),'image_size':(32,32),'detection_scores':torch.ones(1),'reid_features':torch.tensor([[1.,0.]])}
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'trace.jsonl';events=[]
        for frame in (1,2):
            for q,legal,off in [('MATCH_DECISION',['ACCEPT_CURRENT','START_NEW'],'START_NEW' if frame==2 else 'ACCEPT_CURRENT'),('MEMORY_DECISION',['WRITE_MEMORY','SKIP_MEMORY'],'WRITE_MEMORY')]:
                events.append({'question':q,'legal_actions':legal,'off_action':off,'state_feature_vector':[0.]*64,'context':{'video_id':1,'frame':frame,'view':0,'decision_scope':'match' if q=='MATCH_DECISION' else 'memory','detection_index':0,'proposal_track_id':1,'track_id':1}})
        path.write_text(''.join(json.dumps(e)+'\n' for e in events))
        cache=Cache();bundle=({1:{'file_name':'fixture'}},{(1,1,f):f for f in range(3)},{f:[(7,[0,0,10,10])] for f in range(3)},{f:{'width':32,'height':32} for f in range(3)})
        kwargs={'trace':path,'cache_root':Path(directory),'annotations':Path(directory),'checkpoint_hash':'fixture','horizon':1,'association_backend':'cosine_contract','cache_obj':cache,'gt_bundle':bundle}
        all_rows,_,_=build_v2_records(**kwargs)
        observed=[]
        rows,_,_=build_v2_records(**kwargs,emit_event_filter=lambda e:e['_frame']==1 and e['question']=='MEMORY_DECISION',rollout_observer=lambda e,a,s,h:observed.append(s[-1]['actions']))
        assert len(rows)==1
        assert rows[0]==next(r for r in all_rows if r['frame']==1 and r['question_type']=='MEMORY_DECISION')
        assert observed and all(x[0]=='START_NEW' for x in observed),observed
    print('Phase V GT coordinates and full-future emission filter: PASS')
def test_phase5_diagnostics():
    main()

if __name__=='__main__':main()

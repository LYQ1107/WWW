"""Pinned official ByteTrack on available frozen TRAIN boxes, no invented lows."""
import argparse,json,os,sys,time
from types import SimpleNamespace
from jev_phase7_common import *

def run(video):
    protect();dest=OUT/'bytetrack'/f'video{video:02d}'
    if dest.exists():raise RuntimeError('refusing reused ByteTrack output')
    dest.mkdir(parents=True)
    root=OUT/'external_sources/FoundationVision__ByteTrack'
    for p in (root/'yolox',root/'yolox/tracker'):
        p.mkdir(parents=True,exist_ok=True);(p/'__init__.py').write_text('')
    sys.path.insert(0,str(root));sys.path.insert(0,str(OUT/'isolated_dependencies'))
    import numpy as np
    if 'float'not in np.__dict__:np.float=float # compatibility alias only; official tracker unmodified
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracker.basetrack import BaseTrack
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    empty=OUT/'empty_debug_inputs.jsonl';empty.write_text('')if not empty.exists()else None
    os.environ.update(JEV_VIDEO_ID=str(video),JEV_TRACE_PATH=str(empty),JEV_RECORDS_PATH=str(empty),JEV_PILOT_ROOT=str(dest))
    import run_early_pilot_tracking as pilot
    pilot.VIDEO_ID=video;pilot.TRACE=pilot.RECORDS=empty;pilot.PILOT=dest
    _,subset,lookup,_,_=pilot.load_inputs();cache=FrozenPerceptionCache(CACHE)
    args=SimpleNamespace(track_thresh=.5,match_thresh=.8,track_buffer=30,mot20=False)
    BaseTrack._count=0;trackers={view:BYTETracker(args,frame_rate=30)for view in (0,1)}
    predictions=[];lows=highs=0;stamp=time.perf_counter()
    for key in [k for k in cache.keys()if k[0]==video]:
        payload=cache.load(*key);boxes=payload['pred_boxes'].numpy();scores=payload['detection_scores'].numpy()
        lows+=int(((scores>.1)&(scores<.5)).sum());highs+=int((scores>.5).sum())
        data=np.concatenate((boxes,scores[:,None]),axis=1)
        size=payload['image_size'];tracks=trackers[key[2]].update(data,size,size)
        image=pilot.image_for(lookup,*key)
        for track in tracks:
            predictions.append({'image_id':int(image['id']),'category_id':1,'track_id':int(track.track_id),
                                'bbox':pilot.scale_box(track.tlbr,size,image),'score':float(track.score)})
    wall=time.perf_counter()-stamp;path=dest/'predictions.json';save(path,predictions)
    dataset=pilot.prepare_eval_dataset(subset);prepared,evaluated=pilot.run_eval('bytetrack',path,dataset)
    value={'status':'HIGH_ONLY_AVAILABLE_INPUT'if lows==0 else 'COMPLETE','binding':binding(),'video':video,
           'official_tracker_commit':'d1bf0191adff59bc8fcfeaa0b33d3d1642552a99',
           'official_source_sha256':{str(p.relative_to(root)):sha(p)for p in (root/'yolox/tracker').glob('*.py')if p.name!='__init__.py'},
           'parameters':vars(args),'frame_rate':30,'high_detections':highs,'low_detections':lows,
           'low_stage_status':'NOT_EXERCISED'if lows==0 else 'EXERCISED',
           'metrics':pilot.extract_metrics(evaluated),'predictions':str(path),'predictions_sha256':sha(path),
           'evaluation':str(evaluated),'prepared':str(prepared),'wall_seconds_excluding_eval':wall,
           'camera_trackers':'independent; globally distinct track IDs; no cross-camera fusion',
           'comparison_scope':'traditional tracker on available perception; different Kalman/state/output boxes, not same-solver attribution',
           'fabricated_detections':False,'official_test_read':False}
    save(dest/'result.json',value);protect();print(json.dumps({'video':video,'status':value['status'],'metrics':value['metrics']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);run(p.parse_args().video)

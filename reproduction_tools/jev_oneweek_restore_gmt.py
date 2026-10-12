"""Restore complete original GMT association from its actual 1152D cache.

This is a distinct full-system reference, never a VFCE1024 common-front comparison
and never an image-to-tracks FPS measurement. Full raw online outputs are checked
against immutable Phase XII per-video predictions; no GT enters inference.
"""
import argparse
import collections
import gzip
import traceback
import torch
from jev_oneweek_common import *
from jev_oneweek_count_params import original_model


def main(video,attempt):
    protect();assert video in DEV;seed_all(20261008)
    from jev_phase13_runtime import cache_inputs,run
    from run_jev_phase10_closed_loop import raw_predictions
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    source=read(ROOT/'reports/JEV_PHASE12/MATCH_VALIDATION_RESULTS.json')
    group=next(c for c in source['cases'] if c['case']['kind']=='off')
    old=next(c for c in group['per_video'] if c['video']==video)
    original_predictions=old['strict_predictions']
    assert sha(original_predictions['path'])==original_predictions['SHA256']
    folder=OUT/'original_GMT_restore'/f'video{video:02d}'/f'attempt{attempt:02d}'
    folder.mkdir(parents=True,exist_ok=False)
    assert not (folder/'RESULT.json').exists(),'Complete runs immutable'
    values,frames,_=cache_inputs(video);assert frames==old['frames']
    model,cfg=original_model()
    cache=Path('/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train')
    model.jev_perception_cache_reader=FrozenPerceptionCache(cache)
    assert model.jev_stage2_executor is None and model.jev_candidate_policy is None
    counts=collections.Counter();begin=time.monotonic();last=[0.]
    journal=gzip.open(folder/'COMMITS.jsonl.gz','wt')
    def committed(**d):
        key=[video,int(d['frame']),int(d['view'])]
        ids=d['instances'][-1].track_ids.cpu().tolist()
        counts['payloads']+=1;counts['detections']+=len(ids)
        journal.write(json.dumps(dict(key=key,ids=ids))+'\n')
        if time.monotonic()-last[0]>10:
            journal.flush()
            save(folder/'PROGRESS.json',dict(status='RUNNING',key=key,frames=frames,
                                           counts=dict(counts),elapsed=time.monotonic()-begin))
            print('ORIGINAL_GMT_PROGRESS',key,frames,flush=True);last[0]=time.monotonic()
    model.jev_candidate_commit_observer=committed
    save(folder/'START.json',dict(binding=binding('Original complete GMT cache-backed native cold start'),
                                 original_predictions=original_predictions,frames=frames,
                                 original_cache_index=ref(cache/'index.jsonl'),GT_actor_inputs=False))
    try:
        with torch.no_grad():raw,_=run(model,values,frames)
    finally:journal.close()
    # Metadata has image IDs and output scale only, no identity labels.
    from jev_phase13_common import ANNOTATIONS
    images={(i['frame_id']-1,i['view_id']-1):i for i in read(ANNOTATIONS)['images'] if i['video_id']==video}
    predictions=raw_predictions(raw,images)
    save(folder/'RAW_PREDICTIONS.json',predictions)
    expected=read(original_predictions['path'])
    differences=[]
    for index,(a,b) in enumerate(zip(predictions,expected)):
        if a!=b:
            differences.append(dict(row=index,actual=a,expected=b))
            if len(differences)>=8:break
    exact=len(predictions)==len(expected) and not differences
    result=dict(status='PASS_EXACT_ORIGINAL_GMT_RESTORATION' if exact else 'FAIL_PARITY_ORIGINAL_RESULTS_PRESERVED',
                binding=binding('Complete real native mutated-state replay; no new GT evaluation'),
                video=video,frames=frames,counts=dict(counts),elapsed_seconds=time.monotonic()-begin,
                predictions=ref(folder/'RAW_PREDICTIONS.json'),original=original_predictions,
                actual_rows=len(predictions),expected_rows=len(expected),raw_fields_and_global_IDs_exact=exact,
                diagnostic_first_differences=differences,actual_mutated_state_online=True,
                original_RPCE1152=True,common_Stage1_VFCE1024=False,
                full_scene_FPS=None,efficiency_scope='Cached perception, not full live image execution',
                source_checkpoint=ref('/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth'),
                official_TEST=False,sealed20_21_22=True)
    save(folder/'RESULT.json',result);save(REPORTS/f'ORIGINAL_GMT_RESTORE_VIDEO{video:02d}.json',result)
    print(result['status'],video,flush=True)
    if not exact:raise RuntimeError('Original GMT parity mismatch, retained for diagnosis')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--video',type=int,required=True);args=parser.parse_args()
    try:main(args.video,1)
    except Exception:
        failure('original_GMT_restore',traceback.format_exc());raise

"""Verify native/Prior detector and retrieval cohorts on completed fresh runs.

Reuse the existing post-run evaluator and paired-query definitions. No actor,
reranking or metric overwrite; all-request outcomes and eligibility drift stay
separate from conditional Recall and from tracking solver/final-ID corrections.
"""
import time
import collections
import numpy as np
from common import *
from audit_retrieval_outcomes import load,compare


def verify_all_detector_requests(info,outcomes,root,data):
    expected=collections.Counter();images=0
    for scene in info['video_ids']:
        records=video_records(data,scene);images+=len(records)
        ordinal={frame:i for i,frame in enumerate(sorted({r['frame_id'] for r in records}))}
        metadata={r['image_id']:r for r in records}
        for row in json.loads((root/f'video_{scene:02d}.json').read_text()):
            meta=metadata[row['image_id']];step=ordinal[meta['frame_id']]
            if step%info['query_interval']:continue
            key=(scene,meta['view_id'],step,row['image_id'],meta['frame_id'],row['track_id'],tuple(row['bbox']))
            expected[key]+=1
    actual=collections.Counter()
    for row in outcomes:
        q=row['query'];box=np.asarray(q['bbox'],dtype=np.float32).copy()
        # Same float32 XYXY->XYWH subtraction as the unchanged native COCO
        # exporter, avoiding an inexact reverse transform of serialized widths.
        box[2:]-=box[:2]
        key=(q['scene'],q['camera'],q['round_id'],q['image_id'],q['frame_id'],row['published_track_id'],tuple(box.tolist()))
        actual[key]+=1
    if images!=info['images'] or sum(expected.values())!=info['requests'] or actual!=expected:
        raise ValueError('actual requests are not exactly every native detector output at the fixed causal query interval')
    return dict(images=images,requests=sum(expected.values()),
        exact_all_detector_query_membership_with_multiplicity=True,
        scope='Metadata ordinals and completed detector outputs only; no GT-based selection, query repetition or silent omission.')


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')
    directory=runtime/'diagnostics/system_query_outcomes_v1';directory.mkdir(parents=True,exist_ok=True)
    loaded={};artifacts={};pending=[];created=[];retention={};start=time.perf_counter();data=train_data()
    for streams in [2,4,10]:
        for method in ['off','prior']:
            tag=f'trackvggt_runtime_{method}_{streams}streams_full_v1';root=runtime/'retrieval'/tag
            manifest=root/'manifest.json';metric=root/'retrieval.metrics.json';parity=runtime/'matrix_reports'/(tag+'_parity.json')
            if not (manifest.exists() and metric.exists() and parity.exists()):pending.append(tag);continue
            info=json.loads(manifest.read_text());proof=json.loads(parity.read_text())
            if not info['full_video'] or proof['status']!='PASS' or not proof['full_video']:
                raise ValueError('paired system analysis needs verified complete native runs')
            if proof['rows'][0]['manifest_sha256']!=sha256(manifest):raise ValueError('full parity bound to another producer')
            target=directory/(tag+'.json')
            if not target.exists():
                stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                log=runtime/'logs'/(tag+'_verify_outcomes_'+stamp+'.log')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                begun=time.perf_counter()
                with log.open('x') as stream:
                    subprocess.run([sys.executable,'geo_tools/evaluate_live_retrieval.py','--run',str(root),
                        '--verify-existing','--outcome-output',str(target)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                created.append(dict(tag=tag,CPU_evaluation_seconds=time.perf_counter()-begun,log=str(log),log_sha256=sha256(log)))
            outcomes,metrics=load(target)
            retention[tag]=verify_all_detector_requests(info,outcomes,root,data)
            loaded[streams,method]=(outcomes,metrics,info)
            artifacts.update({str(target):sha256(target),str(metric):sha256(metric),str(manifest):sha256(manifest),str(parity):sha256(parity)})
    rows=[]
    for streams in [2,4,10]:
        if (streams,'off') not in loaded or (streams,'prior') not in loaded:continue
        baseline,bmetric,binfo=loaded[streams,'off'];prior,pmetric,pinfo=loaded[streams,'prior']
        if binfo['video_ids']!=pinfo['video_ids']:raise ValueError('paired scenes differ')
        for field in ['seed','checkpoint_sha256','config_sha256','annotation_sha256','query_interval','index']:
            if field=='index':
                for key in ['clip_groups','ttl_groups','feature_age','max_payload_bytes']:
                    if binfo[field][key]!=pinfo[field][key]:raise ValueError('paired native index settings differ')
            elif binfo[field]!=pinfo[field]:raise ValueError('paired native inputs differ: '+field)
        scenes=[]
        for scene in binfo['video_ids']:
            paths=[runtime/'retrieval'/f'trackvggt_runtime_{method}_{streams}streams_full_v1'/f'video_{scene:02d}.json' for method in ['off','prior']]
            values=[json.loads(path.read_text()) for path in paths]
            stripped=[[{k:v for k,v in row.items() if k!='track_id'} for row in value] for value in values]
            if stripped[0]!=stripped[1]:raise ValueError('OFF/Prior detector membership/box/score/order differs')
            scene_result=compare([r for r in baseline if r['query']['scene']==scene],[r for r in prior if r['query']['scene']==scene])
            scenes.append(dict(scene=scene,full_detector_rows=len(values[0]),identical_full_detector_boxes_scores_membership_and_order=True,**scene_result))
            artifacts.update({str(path):sha256(path) for path in paths})
        rows.append(dict(configured_streams=streams,**compare(baseline,prior),
            baseline_conditional_Recall1=bmetric['recall_at_k']['1'],prior_conditional_Recall1=pmetric['recall_at_k']['1'],
            baseline_success1_all_requests=bmetric['success_at_k_all_requests']['1'],prior_success1_all_requests=pmetric['success_at_k_all_requests']['1'],
            per_scene=scenes))
    report=dict(status='COMPLETE_ACTUAL_NATIVE_QUERY_PAIRS' if len(rows)==3 and not pending else 'PARTIAL_ACTUAL_NATIVE_QUERY_PAIRS',
        rows=rows,pending=pending,verified_complete_outcome_runs=len(loaded),new_postrun_CPU_evaluations=created,
        current_CPU_analysis_seconds=time.perf_counter()-start,source_artifacts=artifacts,all_detector_request_retention=retention,
        scope='Actual full native/Prior outputs; detector boxes/scores/membership/order identical after removing only track_id. Query occurrences/GT labels exactly paired; common eligibility, gains/losses, rank fixes/harms and all-request success separate.',
        limitations='Offline GT evaluation only after inference, no metric overwrite or reranking. These are retrieval response rank changes, not primary Hungarian/Bank/final tracking corrections or a VGGT method benefit. CPU evaluator time is not actor FPS.',
        scientific_success=False,provenance=provenance())
    save_json(ROOT/'results/trackvggt_system_query_pairs_v1.json',report)
    print(json.dumps(dict(status=report['status'],completed_pairs=len(rows),pending=len(pending),
        comparisons=[dict(streams=r['configured_streams'],requests=r['requests'],common_eligible=r['common_eligible'],
            fixes=r['common_eligible_rank1_fixes'],harms=r['common_eligible_rank1_harms'],eligible_gained=r['eligible_gained'],eligible_lost=r['eligible_lost']) for r in rows])))


if __name__=='__main__':main()

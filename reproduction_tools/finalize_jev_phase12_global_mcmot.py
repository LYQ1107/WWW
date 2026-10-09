"""GT-only joint-camera scene evaluation; never imported by the online actor."""
import argparse,time,subprocess,os,collections
from jev_phase12_common import *

def metric(predictions,out,scope):
    ann=json.loads(Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json').read_text());images={i['id']:i for i in ann['images'] if i['video_id'] in VAL};groups=collections.defaultdict(list);lengths={};sources={}
    prepared=out/'prepared';gtroot=prepared/'gt';tracker=prepared/'trackers/GMT/data';tracker.mkdir(parents=True,exist_ok=True)
    for image in images.values():
        seq=image['file_name'].split('/',1)[0];scene=seq.rsplit('_View',1)[0];lengths[scene]=max(lengths.get(scene,0),int(image['frame_id'])*2)
        sources[seq]=Path('/data/DATASETS/TRACKING/JDE/VisionTrack/train')/seq/'gt/gt.txt'
    gt=collections.defaultdict(list)
    for seq,path in sorted(sources.items()):
        scene,view=seq.rsplit('_View',1);camera=int(view)-1;assert camera in [0,1]
        for line in path.read_text().splitlines():
            if not line.strip():continue
            fields=line.split(',');fields[0]=str((int(float(fields[0]))-1)*2+camera+1);gt[scene].append(fields)
    for scene,rows in gt.items():
        rows.sort(key=lambda f:(int(f[0]),int(float(f[1]))));path=gtroot/scene/'gt/gt.txt';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(','.join(f)+'\n' for f in rows))
        assert len({(f[0],f[1]) for f in rows})==len(rows),'duplicate IDs in actual joint-camera GT'
    for p in predictions:
        image=images[p['image_id']];seq=image['file_name'].split('/',1)[0];scene=seq.rsplit('_View',1)[0];frame=(int(image['frame_id'])-1)*2+int(image['view_id']);x,y,w,h=p['bbox']
        groups[scene].append((frame,int(p['track_id']),x,y,w,h,float(p['score']),-1,-1,-1))
    for scene,rows in groups.items():
        rows.sort(key=lambda r:(r[0],r[1]));(tracker/(scene+'.txt')).write_text(''.join(','.join(map(str,r))+'\n' for r in rows))
    manifest={'status':'PASS','trackeval_gt':str(gtroot),'trackeval_trackers':str(prepared/'trackers'),'tracker_name':'GMT','seq_lengths':lengths,'gt_duplicate_rows':{scene:0 for scene in lengths},'GT_sources_SHA256':{seq:sha(path) for seq,path in sources.items()},'identity_scope':'same official identity IDs shared across two views within each scene; no remap between cameras','time_definition':'virtual index 2*(frame-1)+(view-1)+1; per-view spatial matching, shared ID association across cameras; camera-frame CLEAR counts, no physical FPS interpretation','scope':scope,'strict_online':scope=='strict_online','future_min_track_length_filter':scope!='strict_online'}
    save(prepared/'manifest.json',manifest);env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run([PYTHON,str(ROOT/'reproduction_tools/evaluate_visiontrack.py'),'--prepared',str(prepared),'--output',str(out/'evaluation')],cwd=ROOT,env=env,check=True)
    from run_early_pilot_tracking import extract_metrics
    return {'metrics':extract_metrics(out/'evaluation'),'manifest':manifest,'combined_metrics_SHA256':sha(out/'evaluation/metrics.json')}

def main(partial):
    protect();protocol=json.loads((REPORTS/'ONLINE_PROTOCOL.json').read_text());completed=[]
    for case in protocol['cases']:
        runs=[]
        for video in VAL:
            path=OUT/'validation_closed_loop_v1'/case['name']/f'video{video:02d}/RESULT.json'
            if not path.exists():break
            run=json.loads(path.read_text());assert run['status']=='COMPLETE' and run['case']==case;runs.append(run)
        if len(runs)!=3:continue
        out=OUT/'joint_camera_validation_v1'/case['name'];out.mkdir(parents=True,exist_ok=True);done=out/'RESULT.json'
        if done.exists():completed.append(json.loads(done.read_text()));continue
        attempt=0
        while (out/f'attempt{attempt:03d}').exists():attempt+=1
        folder=out/f'attempt{attempt:03d}';result={'status':'COMPLETE','case':case,'metrics':{}}
        for scope,field in [('strict_online','strict_predictions'),('canonical_GMT_filtered','canonical_predictions')]:
            predictions=[]
            for run in runs:assert sha(run[field]['path'])==run[field]['SHA256'];predictions.extend(json.loads(Path(run[field]['path']).read_text()))
            result['metrics'][scope]=metric(predictions,folder/scope,scope)
        save(done,result);completed.append(result);print('JOINT_CAMERA_MCMOT_CASE_COMPLETE',case['name'],flush=True)
    complete=len(completed)==len(protocol['cases']);save(OUT/'JOINT_CAMERA_PROGRESS.json',{'status':'COMPLETE' if complete else 'RUNNING','cases':len(completed),'total':len(protocol['cases'])})
    if complete:save(REPORTS/'GLOBAL_MCMOT_VALIDATION.json',{'status':'COMPLETE','cases':completed,'scope':'supplementary jointly associated scene HOTA/AssA/IDF1 across both cameras; raw official GT IDs preserved, cameras interleaved as virtual per-camera timesteps; differs from standard six-camera-sequence TrackEval and never substitutes actor inference','heldout':'SEALED','Full24':False,'official_TEST':False})
    else:assert partial
    return complete
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        done=main(a.partial or a.watch)
        if done or not a.watch:break
        time.sleep(30)

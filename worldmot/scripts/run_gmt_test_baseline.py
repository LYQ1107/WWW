#!/usr/bin/env python3
"""Separate frozen native-GMT comparator; WorldMOT never imports this module.

No CIR/geometry intervention and no short-track future-count filtering. TEST
uses metadata-only JSON. This does not bypass the old TRAIN-only CIR runner:
it is a separate, explicitly locked baseline entry for this experiment.
"""
import argparse,collections,hashlib,json,os,sys,time
from pathlib import Path
from types import MethodType
from experiment_provenance import source_stamp
from preserve_inference_attempts import prepare_scene

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='/data1/liuyeqiang/WWW_cir_gmt_runtime_20261016/sources/p2_full_controls_v2')
    p.add_argument('--metadata',required=True);p.add_argument('--split',choices=['train','test'],required=True);p.add_argument('--output',required=True);p.add_argument('--scenes',nargs='+',type=int);p.add_argument('--limit',type=int)
    a=p.parse_args();source=Path(a.source);sys.path[:0]=[str(source/'geo_tools'),str(source),str(source/'third_party/CenterNet2')]
    import torch;torch.set_num_threads(2)
    from common import load_gmt,seed_all,CHECKPOINT,SOURCE_HASHES,SOURCE_COMMIT,SOURCE_DIFF_SHA256
    from run_cir_online import CurrentImages
    from test_cir_native_commit import original_method
    from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
    data=json.loads(Path(a.metadata).read_text())
    if data.get('annotations'):raise ValueError('GT-bearing inference metadata forbidden')
    if a.split=='test' and a.limit:raise ValueError('TEST must run complete videos')
    cfg,model=load_gmt(20261014,{'ENABLED':False});cfg.defrost();cfg.MODEL.CIR.ENABLED=False;cfg.MODEL.CIR.MODE='off'
    model.cir_controller.enabled=False;model.cir_controller.mode='off';model.run_global_tracker_plus=MethodType(original_method(),model)
    model.min_track_len=0;model.eval().requires_grad_(False)
    if model.roi_heads.delay_cls:raise RuntimeError('future-dependent delayed classification is forbidden for this causal comparison')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    lock={'checkpoint_sha256':sha(CHECKPOINT),'config_sha256':hashlib.sha256(cfg.dump().encode()).hexdigest(),'split':a.split,'native_method':'original run_global_tracker_plus from frozen base f595189e62dac917a4880b1220b3c9ae6d7303aa',
          'CIR_enabled':False,'GEO_enabled':False,'future_short_track_filter':False,'future_delayed_classification':False,
          'native_source_commit':SOURCE_COMMIT,'native_source_diff_sha256':SOURCE_DIFF_SHA256,'native_source_file_sha256':SOURCE_HASHES,
          'source':str(source),'metadata_sha256':sha(a.metadata),'GT_in_inference':False,'source_sha256':source_stamp()['sha256']}
    (root/'GMT_FROZEN_LOCK.json').write_text(json.dumps(lock,indent=2))
    for sid in a.scenes or [v['id'] for v in data['videos']]:
        out=root/f'scene_{sid:02d}'
        signature={k:lock[k] for k in ('checkpoint_sha256','config_sha256','split','metadata_sha256','source_sha256')}
        if not prepare_scene(out,signature):continue
        records=sorted([dict(x,image_id=x['id'],file_name=str(Path('/data/DATASETS/TRACKING/JDE/VisionTrack')/a.split/x['file_name']),view_num=2) for x in data['images'] if x['video_id']==sid and (not a.limit or x['frame_id']<=a.limit)],key=lambda x:(x['view_id'],x['frame_id']))
        batch=CurrentImages(cfg,records);seed_all(20261014)
        (out/'manifest.json').write_text(json.dumps({'status':'RUNNING',**lock,'scene':sid,'images_expected':len(records)},indent=2))
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():outputs,views=model(batch)
        torch.cuda.synchronize();forward_s=time.perf_counter()-started
        ordered=sorted(records,key=lambda x:(x['frame_id'],x['view_id']))
        if len(outputs)!=len(records) or set(batch.processed)!={r['image_id'] for r in records}:raise RuntimeError('incomplete native GMT video')
        groups=collections.OrderedDict()
        for rec,result in zip(ordered,outputs):
            timestamp=rec['frame_id'];row=groups.setdefault(timestamp,{'scene_id':sid,'timestamp':timestamp,'tracks_2d_by_camera':{'1':[],'2':[]},'image_ids':{},'causal':True,'native_GMT':True})
            row['image_ids'][str(rec['view_id'])]=rec['image_id']
            for obj in custom_instances_to_coco_json(result['instances'].to('cpu'),rec['image_id']):
                row['tracks_2d_by_camera'][str(rec['view_id'])].append({'global_id':int(obj['track_id']),'bbox_xywh':obj['bbox'],'score':float(obj['score']),'visible':True,'evaluation_eligible':True,'output_source':'NATIVE_GMT_RAW_CAUSAL'})
        with (out/'tracks.jsonl').open('w') as f:
            for row in groups.values():f.write(json.dumps(row)+'\n');f.flush()
        e2e=time.perf_counter()-started
        report={'status':'COMPLETE',**lock,'scene':sid,'frames':len(groups),'images':len(records),'full_video':not bool(a.limit),'actual_inference_forward_s':forward_s,
                'real_e2e_seconds':e2e,'real_e2e_fps':len(groups)/e2e,'fps_unit':'synchronized dual-camera timestamps/second',
                'timing_scope':'RGB read/map + frozen native detector/descriptor/GTA/Bank + native postprocess + serialization and final full-video ID delivery; excludes initialization and evaluation',
                'delivery_latency_scope':'native model emits postprocessed Instances at complete-video return; per-timestamp final delivery latency not claimed',
                'prediction_sha256':sha(out/'tracks.jsonl'),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__}
        (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({'scene':sid,'frames':len(groups),'real_e2e_fps':report['real_e2e_fps']}),flush=True)
if __name__=='__main__':main()

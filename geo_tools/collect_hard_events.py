"""Read-only GMT association audit. Select current ambiguity without identity labels."""
import argparse,heapq,time,numpy as np,torch
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--video-id',type=int,default=18);p.add_argument('--limit',type=int,default=12);p.add_argument('--tag',default='hard_events_v1');a=p.parse_args()
    out=RUNTIME/'hard_events'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);cfg,model=load_gmt();data=train_data();records=video_records(data,a.video_id);record_index={(r['view_id'],r['frame_id']):r for r in records};heap=[];counter=0
    original_infer=model.inference;original_asso=model.get_asso
    def infer(batch,views,times,view,**kw):
        result=original_infer(batch,views,times,view,**kw)
        for inst in result:
            n=len(inst);device=inst.reid_features.device
            # Uniform observation-only fields survive the original Instances.cat.
            inst.audit_camera=torch.full((n,),int(batch[0]['view_id']),device=device,dtype=torch.long)
            inst.audit_frame=torch.full((n,),int(Path(batch[0]['file_name']).stem[-6:]),device=device,dtype=torch.long)
        return result
    def association(instances,k,**kw):
        nonlocal counter
        result=original_asso(instances,k,**kw)
        if kw.get('geo_context',True) is False:return result
        query=instances[k];refs=[x for i,x in enumerate(instances) if i!=k]
        if len(query)==0 or not query.has('audit_camera') or any(not r.has('audit_camera') for r in refs):return result
        if int(query.audit_frame[0])==1:return result
        logits=result[0][0];ids=torch.cat([x.track_ids for x in refs]);cams=torch.cat([x.audit_camera for x in refs]);frames=torch.cat([x.audit_frame for x in refs])
        unique=ids.unique();ind=(ids[:,None]==unique[None,:]).float()
        # Original global tracker activates each historical frame block separately.
        blocks=logits.split([len(x) for x in refs],dim=1)
        prob=torch.cat(model.roi_heads._activate_asso(blocks),dim=1);score=prob@ind
        if score.shape[1]<2:return result
        top=score.topk(2,dim=1);margins=(top.values[:,0]-top.values[:,1])/(top.values[:,0].abs()+1e-8)
        for qi in range(len(query)):
            selected=top.indices[qi];legal=torch.isin(ids,unique[selected])&(cams!=query.audit_camera[qi])
            if not legal.any():continue
            # Choose at most four most recent opposite-camera target proposals per competing track.
            columns=[]
            for track in unique[selected]:
                candidates=((ids==track)&(cams!=query.audit_camera[qi])).nonzero(as_tuple=True)[0]
                if len(candidates):columns.extend(candidates[frames[candidates].argsort(descending=True)[:4]].tolist())
            columns=torch.tensor(columns,device=ids.device);priority=float(1-margins[qi]);counter+=1
            if len(heap)>=a.limit and priority<=heap[0][0]:continue
            qbox=query.pred_boxes.tensor[qi].detach().cpu().numpy();h,w=query.image_size
            # Convert separately using each source view record, keeping original detector postprocess scale convention.
            qcam=int(query.audit_camera[qi]);qframe=int(query.audit_frame[qi]);qrecord=record_index[(qcam,qframe)]
            qbox=qbox*np.array([qrecord['width']/w,qrecord['height']/h]*2)
            allboxes=torch.cat([x.pred_boxes.tensor for x in refs]);sizes=[x.image_size for x in refs for _ in range(len(x))]
            candidate_rows=[];all_features=torch.cat([x.reid_features for x in refs])
            for ci in columns.tolist():
                cam,frame=int(cams[ci]),int(frames[ci]);record=record_index[(cam,frame)];hh,ww=sizes[ci]
                box=allboxes[ci].detach().cpu().numpy()*np.array([record['width']/ww,record['height']/hh]*2)
                candidate_rows.append(dict(camera_id=cam,frame_id=frame,image_id=record['image_id'],box=box.tolist(),track_id=int(ids[ci]),gmt_logit=float(logits[qi,ci]),gmt_probability=float(prob[qi,ci]),gmt_feature=all_features[ci].detach().cpu().tolist()))
            event=dict(video_id=a.video_id,query_camera=qcam,query_frame=qframe,query_image_id=qrecord['image_id'],query_box=qbox.tolist(),
                query_gmt_feature=query.reid_features[qi].detach().cpu().tolist(),
                score_semantics='per-frame unmatched softmax followed by original GTA sum',relative_gta_margin=float(margins[qi]),candidate_tracks=unique[selected].cpu().tolist(),candidate_track_scores=top.values[qi].cpu().tolist(),candidates=candidate_rows,
                sampling='lowest current relative GTA margin with opposite-camera candidates; no GT labels read')
            item=(priority,counter,event)
            if len(heap)<a.limit:heapq.heappush(heap,item)
            else:heapq.heapreplace(heap,item)
        return result
    model.inference=infer;model.get_asso=association;batch=map_records(cfg,records);seed_all(20261014)
    with torch.inference_mode():outputs,_=model(batch)
    events=[x[2] for x in sorted(heap,reverse=True)]
    save_json(out/'events.json',dict(provenance=provenance(),events=events,complete_video=True,gt_used_for_selection=False))
    from gtr.evaluation.mot_evaluation import align_visiontrack_inputs_to_outputs,custom_instances_to_coco_json
    rows=[]
    for record,output in zip(align_visiontrack_inputs_to_outputs('VISION_train',records,outputs),outputs):rows.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),record['image_id']))
    save_json(out/'predictions.json',rows)
    print(json.dumps(dict(events=len(events),predictions_sha256=sha256(out/'predictions.json')),indent=2))
if __name__=='__main__':main()

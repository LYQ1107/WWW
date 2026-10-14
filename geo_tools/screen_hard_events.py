"""Pair-level identity diagnostics on native-GTA ambiguity, using original detections.

This is an offline diagnostic, not closed-loop tracking accuracy. Feature groups
use only simultaneous views of a source timestamp; no independent 3D comparison.
"""
import argparse,collections,csv,time,numpy as np,torch
from common import *
from feature_metrics import evaluate_scores,normalize

from target_labels import identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--events',type=Path,required=True);p.add_argument('--tag',default='hard_gta_features_v1');p.add_argument('--dim',type=int,default=256);a=p.parse_args()
    source=json.loads(a.events.read_text());events=source['events']
    if not events:raise ValueError('no events')
    if any(e.get('score_semantics')!='per-frame unmatched softmax followed by original GTA sum' for e in events):raise ValueError('wrong native GMT score semantics')
    split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text());vids={e['video_id'] for e in events}
    if not vids.issubset(set(split['strict_development_video_ids'])):raise ValueError('this diagnostic accepts clean TRAIN development only')
    out=RUNTIME/'hard_features'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);data=train_data();annotations=collections.defaultdict(list)
    for ann in data['annotations']:annotations[ann['image_id']].append(ann)
    records={r['image_id']:r for vid in vids for r in video_records(data,vid)}
    groups=collections.defaultdict(dict)
    def key(image_id,box):return (image_id,tuple(float(x) for x in box))
    for e in events:
        items=[(e['query_image_id'],e['query_box'])]+[(c['image_id'],c['box']) for c in e['candidates']]
        for image_id,box in items:
            r=records[image_id];groups[(r['video_id'],r['frame_id'])][key(image_id,box)]=(r,box)
    sys.path[:0]=[str(RUNTIME/'sources/vggt'),str(RUNTIME/'sources/dinov2')]
    from vggt.utils.load_fn import load_and_preprocess_images
    from gtr.geometry.roi_feature_extractor import PadMapping
    from gtr.geometry.vggt_adapter import FrozenVGGT
    from gtr.geometry.dinov2_adapter import FrozenDINOv2
    seed_all(20261014);vggt=FrozenVGGT(RUNTIME/'weights/vggt_1b.pt');features={};diagnostics=[]
    frame_index={(r['video_id'],r['frame_id'],r['view_id']):r for r in records.values()}
    for (vid,frame),items in sorted(groups.items()):
        rows=[frame_index[(vid,frame,v)] for v in [1,2]];view_items=[[x for x in items.items() if x[1][0]['view_id']==v] for v in [1,2]]
        boxes=[torch.tensor([x[1][1] for x in vs],dtype=torch.float32).reshape(-1,4) for vs in view_items]
        mapped=[PadMapping(r['width'],r['height']).boxes(b) for r,b in zip(rows,boxes)]
        t=time.perf_counter();images=load_and_preprocess_images([r['file_name'] for r in rows],mode='pad')
        result,coverage,_=vggt.extract(images,mapped,dim=a.dim);reverse,_,_=vggt.extract(images.flip(0),mapped[::-1],dim=a.dim)
        for name,vs in result.items():
            for v,entries in enumerate(view_items):
                for i,(k,_) in enumerate(entries):features.setdefault(k,{})[f'vggt/{name}']=torch.cat([vs[v][i],reverse[name][1-v][i]])/2**.5
        diagnostics.append(dict(video_id=vid,frame_id=frame,targets=sum(len(v) for v in view_items),zero_patch_targets=sum(int((c==0).sum()) for c in coverage.values()),vggt_calls=2,seconds=time.perf_counter()-t))
        print(json.dumps(diagnostics[-1]),flush=True)
    del vggt;torch.cuda.empty_cache();dino=FrozenDINOv2(RUNTIME/'weights/dinov2_vitl14_reg4.pth')
    for (vid,frame),items in sorted(groups.items()):
        rows=[frame_index[(vid,frame,v)] for v in [1,2]];view_items=[[x for x in items.items() if x[1][0]['view_id']==v] for v in [1,2]]
        boxes=[torch.tensor([x[1][1] for x in vs],dtype=torch.float32).reshape(-1,4) for vs in view_items];mapped=[PadMapping(r['width'],r['height']).boxes(b) for r,b in zip(rows,boxes)]
        images=load_and_preprocess_images([r['file_name'] for r in rows],mode='pad');result,_,_=dino.extract(images,mapped,modes=('mean','center','roi_align'),dim=a.dim)
        for name,vs in result.items():
            for v,entries in enumerate(view_items):
                for i,(k,_) in enumerate(entries):features[k][f'dinov2/{name}']=vs[v][i]
    metrics=collections.defaultdict(list);event_rows=[]
    for number,e in enumerate(events):
        q=key(e['query_image_id'],e['query_box']);refs=[key(c['image_id'],c['box']) for c in e['candidates']]
        qid=identity(e['query_box'],annotations[e['query_image_id']]);ids=np.array([identity(c['box'],annotations[c['image_id']]) for c in e['candidates']]);base=np.array([[c['gmt_logit'] for c in e['candidates']]])
        scores={'gmt_proposal_logits':base,'gmt_reid_cosine':normalize(np.array([e['query_gmt_feature']]))@normalize(np.array([c['gmt_feature'] for c in e['candidates']])).T}
        for name,vector in features[q].items():scores[name]=np.array([[float(vector@features[r][name]) for r in refs]])
        for name,score in scores.items():metrics[name].append(dict(score=score,ids1=ids,ids2=np.array([qid]),base=base))
        event_rows.append(dict(event=number,query_frame=e['query_frame'],query_camera=e['query_camera'],query_gt_identity=qid,candidate_gt_identities=ids.tolist(),native_gta_margin=e['relative_gta_margin'],native_candidate_tracks=e['candidate_tracks']))
    rows=[dict(setting=k,**evaluate_scores(v)) for k,v in metrics.items()]
    report_name='hard_feature_native' if a.dim==0 else 'hard_feature_screening'
    with (ROOT/f'results/{report_name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    save_json(out/'event_labels.json',event_rows);save_json(out/'metrics.json',rows)
    save_json(ROOT/f'results/{report_name}_manifest.json',dict(provenance=provenance(),projection_dim=a.dim,events_sha256=sha256(a.events),event_count=len(events),source_video_ids=sorted(vids),
        diagnostic_scope='selected opposite-camera proposal retrieval only; NOT native full-track GTA or tracking improvement',used_cache=True,cache_scope='selected frozen GMT target boxes/vectors/logits',
        validity_labels_only_after_event_selection=True,source_timestamps=len(groups),diagnostics=diagnostics,vggt_weights_sha256=sha256(RUNTIME/'weights/vggt_1b.pt'),dino_weights_sha256=sha256(RUNTIME/'weights/dinov2_vitl14_reg4.pth')))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()

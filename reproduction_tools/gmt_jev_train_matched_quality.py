"""Complete original/Rule TRAIN trajectories; no GT observer or training."""
import argparse
import gzip
from gmt_jev_runtime import *
from gmt_jev_evaluate import evaluate, duration


def geometry_sha(predictions):
    rows = [{k:v for k,v in row.items() if k != 'track_id'} for row in predictions]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main(video, kind):
    from gtr.modeling.gmt_jev import IdentityCorrection
    from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
    from run_jev_phase10_closed_loop import raw_predictions
    assert video in [12,13,14] and kind in ['original','rule']
    assert read(REPORTS/'BASELINE_PARITY.json')['status'] == 'PASS_EXACT'
    assert read(REPORTS/'BOUNDED_RESIDUAL_ORACLE_AUDIT.json')['status'] == 'COMPLETE_ALL_REGISTERED_TRAIN_OFFLINE_BOUNDED_ORACLE_PROBES'
    folder = OUT/'train_matched_quality'/f'video{video:02d}'/kind
    folder.mkdir(parents=True, exist_ok=False)
    reserve()
    annotation_source = ref(ANNOTATIONS)
    values, frames, images = metadata(video)
    image_hash = hashlib.sha256()
    for image in sorted(images.values(), key=lambda x:x['id']):
        image_hash.update(f"{image['id']}:{sha(IMAGES/image['file_name'])}\n".encode())
    source = binding()
    seed()
    m, cfg, snapshot = model(original=kind=='original')
    actor = IdentityCorrection('rule').cuda().eval() if kind=='rule' else None
    if actor is not None:
        m.gmt_identity_correction = actor
        boot = [False]
        def before(**data):
            if data['first'] and not boot[0]:
                view = 1-data['view']
                actor.committed(data['instances'][view],view,0)
                boot[0] = True
        def committed(**data):
            actor.committed(data['instances'][-1],data['view'],data['frame'])
        m.jev_native_prefix_observer = before
        m.jev_candidate_commit_observer = committed
    seed()
    with torch.no_grad():
        raw, _ = run(m, values, frames)
    predictions = raw_predictions(raw, images)
    stock = m._remove_short_track(list(raw)) if m.min_track_len>0 else list(raw)
    if m.roi_heads.delay_cls:
        stock = m._delay_cls(stock,video_id=video)
    batch = [values[v*frames+f] for f in range(frames) for v in range(2)]
    stock = CustomRCNN._postprocess(stock,batch,[(0,0)]*len(batch),not_clamp_box=m.not_clamp_box)
    stock_predictions = raw_predictions([r['instances'] for r in stock],images)
    paths = {}
    for convention, rows in [('RAW',predictions),('STOCK',stock_predictions)]:
        path = folder/f'{convention}_PREDICTIONS.json.gz'
        with gzip.open(path,'wt') as f:
            json.dump(rows,f)
        paths[convention] = ref(path)
    inference = dict(kind=kind,video=video,frames=frames,predictions=paths,
                     input_images_SHA256=image_hash.hexdigest(),annotation=annotation_source,
                     raw_geometry_and_scores_SHA256=geometry_sha(predictions),
                     effective_config_SHA256=hashlib.sha256(cfg.dump().encode()).hexdigest(),
                     original_snapshot=ref(snapshot) if snapshot else None,
                     full_system_parameters=sum(p.numel() for p in m.parameters()),
                     no_GT_actor_inputs=True,no_gradient_updates=True,no_perception_cache=True,
                     full_image_FPS=None,source=source)
    save(folder/'INFERENCE.json',inference)
    assert sha(ANNOTATIONS) == annotation_source['SHA256'], 'Annotation changed during inference'
    if kind=='rule':
        baseline = read(REPORTS/f'TRAIN_MATCHED_original_VIDEO{video:02d}.json')
        for key in ['input_images_SHA256','effective_config_SHA256','raw_geometry_and_scores_SHA256']:
            assert inference[key] == baseline['inference'][key],key
        assert annotation_source['SHA256'] == baseline['inference']['annotation']['SHA256']
    del m,raw,stock,batch,actor
    gc.collect()
    torch.cuda.empty_cache()
    raw_metrics = evaluate(predictions,folder/'raw_metrics',videos=[video])
    stock_metrics = evaluate(stock_predictions,folder/'stock_metrics',videos=[video])
    assert sha(ANNOTATIONS) == annotation_source['SHA256'], 'Annotation changed during evaluation'
    result = dict(status='COMPLETE_MATCHED_FULL_TRAIN_TRAJECTORY',kind=kind,video=video,
                  inference=inference,primary_raw=raw_metrics,secondary_stock=stock_metrics,
                  primary_identity=duration(predictions,video),secondary_identity=duration(stock_predictions,video),
                  cold_native_mutated_state=True,training_qualification_only=True,
                  independent_validation=False,scientific_success=False,full_image_FPS=None,
                  comparison_scope='Full cold trajectories with same original frontend/seed/native rules; no cross-run numeric-ID or accepted-reference causal claim.',source=binding())
    save(REPORTS/f'TRAIN_MATCHED_{kind}_VIDEO{video:02d}.json',result)
    print('TRAIN_MATCHED_QUALITY_COMPLETE',video,kind,raw_metrics['metrics'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--video',type=int,choices=[12,13,14],required=True)
    p.add_argument('--kind',choices=['original','rule'],required=True)
    a=p.parse_args()
    main(a.video,a.kind)

"""Explicit test_net entry adapter, no legacy hardcoded text writer/eval changes."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from common import CHECKPOINT,sha256,save_json
from run_cir_online import run_loaded,validate_request


def run_spec(cfg,model,spec_path):
    spec_path=Path(spec_path);spec=json.loads(spec_path.read_text())
    if spec['split']!='train':raise ValueError('Official TEST requires the separate final model/parameter lock')
    if len(set(spec['video_ids']))!=len(spec['video_ids']) or not spec['video_ids']:
        raise ValueError('Finite unique registered video list required')
    if sha256(cfg.MODEL.WEIGHTS)!=sha256(CHECKPOINT):raise ValueError('test_net must load the frozen Stage2')
    # Verify what DetectionCheckpointer actually loaded, not merely its filename.
    frozen=torch.load(CHECKPOINT,map_location='cpu')['model'];actual=model.state_dict()
    if set(frozen)!=set(actual) or any(not torch.equal(actual[k].detach().cpu(),v) for k,v in frozen.items()):
        raise ValueError('test_net model tensor differs from frozen GMT checkpoint')
    del frozen,actual
    out=Path(spec['out_root']);jobs=[]
    for video in spec['video_ids']:
        job=SimpleNamespace(video=video,method=spec['method'],qualification=Path(spec['qualification']),
                            out=out/f'video{video}',entrypoint='test_net.py --cir-run-spec')
        validate_request(job);jobs.append(job)
    results={}
    for job in jobs:
        result=run_loaded(job,cfg,model)
        results[str(job.video)]=dict(manifest=str(job.out/'manifest.json'),sha256=sha256(job.out/'manifest.json'),status=result['status'])
    report=dict(status='COMPLETE',split=spec['split'],method=spec['method'],runs=results,
        spec_sha256=sha256(spec_path),actual_loaded_weights_verified=True,
        entrypoint='test_net.py --cir-run-spec -> same model.forward/sliding_inference_GMT; independent output adapter',
        legacy_text_writer_used=False,metric_equations_modified=False,GT_in_inference=False)
    save_json(out/'OFFICIAL_ENTRY_RESULT.json',report)
    return report

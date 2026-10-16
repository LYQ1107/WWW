"""Fresh old-VGGT control on the new fixed pairs, reusing held-out matches."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from omega_common import ROOT, RUNTIME, provenance, save_new, sha256
from omega_modules import geometry_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--omega-run', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261016)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = provenance()
    if source['source_status'].strip():
        raise ValueError('clean immutable source required')
    omega_manifest = args.omega_run/'manifest.json'
    parent = json.loads(omega_manifest.read_text())
    assert parent['status'] == 'BOUNDED_DIAGNOSTIC_COMPLETE'
    assert sha256(parent['cohort']['path']) == parent['cohort']['sha256']
    assert sha256(parent['outputs']['path']) == parent['outputs']['sha256']
    cohort = json.loads(Path(parent['cohort']['path']).read_text())
    pair_map = {p['pair_id']: p for p in cohort['pairs']}
    prior_rows = [json.loads(line) for line in Path(parent['outputs']['path']).read_text().splitlines()]
    out = RUNTIME/'pose_audits'/args.tag
    out.mkdir(parents=True, exist_ok=False)
    old_source = Path('/data1/liuyeqiang/WWW_vggt_geo_runtime/sources/vggt')
    old_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=old_source, text=True).strip()
    assert old_commit == 'a288dd0f14786c93483e45524328726ab7b1b4ce'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=old_source, text=True).strip()
    checkpoint = Path('/data1/liuyeqiang/WWW_vggt_geo_runtime/weights/vggt_1b.pt')
    expected = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
    assert sha256(checkpoint) == expected
    sys.path.insert(0, str(old_source))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    mapping_module = geometry_module('roi_feature_extractor')
    adapter_module = geometry_module('omega_adapter')
    geometry = geometry_module('omega_geometry')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = VGGT().eval().requires_grad_(False)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    state_keys = len(state)
    del state
    model = model.cuda()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-started
    rows = []
    attempted_calls, completed_calls = 0, 0
    with (out/'pair_results.jsonl').open('x') as stream:
        for prior in prior_rows:
            pair = pair_map[prior['pair_id']]
            paths = [i['path'] for i in pair['images']]
            sizes = [(i['original_width'], i['original_height']) for i in pair['images']]
            for image in pair['images']:
                assert sha256(image['path']) == image['sha256']
            corr = prior['correspondence_file']
            assert sha256(corr['path']) == corr['sha256']
            with np.load(corr['path']) as saved:
                eval_p, eval_q = saved['eval_points_A'], saved['eval_points_B']
            row = dict(pair_id=pair['pair_id'], scene=pair['scene'], group=pair['group'],
                       role='OLD_VGGT_FULL_OFFICIAL_FORWARD_CONTROL', original_sizes=sizes, outputs={},
                       eval_matches=len(eval_p), shared_background=prior['correspondence_file'])
            for name, order in [('AB', [0, 1]), ('BA', [1, 0])]:
                t = time.perf_counter()
                try:
                    images = load_and_preprocess_images([paths[i] for i in order], mode='pad')
                    maps = []
                    for i in order:
                        width, height = sizes[i]
                        old_map = mapping_module.PadMapping(width, height)
                        rw, rh = old_map.resized
                        maps.append(adapter_module.PixelMapping(width, height, 0, 0, width, height,
                                    rw, rh, (518-rw)//2, (518-rh)//2, 518, 518))
                    assert tuple(images.shape) == (2, 3, 518, 518)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                        attempted_calls += 1
                        predictions = model(images.cuda())
                    E, K = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
                    torch.cuda.synchronize()
                    completed_calls += 1
                    finite = all(torch.isfinite(v).all().item() for v in predictions.values() if isinstance(v, torch.Tensor))
                    if not finite:
                        raise ValueError('nonfinite full VGGT outputs')
                    pose = geometry.original_fundamental(E[0].float().cpu().numpy(), K[0].float().cpu().numpy(), maps)
                    F = pose['F_original'] if name == 'AB' else pose['F_original'].T
                    item = dict(status='FINITE_INTERFACE_OUTPUTS', F_in_physical_BA=F.tolist(),
                                shapes={k:list(v.shape) for k,v in predictions.items() if isinstance(v, torch.Tensor)},
                                mappings=[m.record() for m in maps],
                                heldout_background=geometry.background_validity(F, eval_p, eval_q, sizes))
                    del predictions
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                    item = dict(status='INVALID_OLD_VGGT_INTERFACE', reason=str(exc), exception_type=type(exc).__name__,
                                heldout_background=geometry.background_validity(None, eval_p, eval_q, sizes))
                item['preprocess_and_forward_seconds'] = time.perf_counter()-t
                row['outputs'][name] = item
            if all('F_in_physical_BA' in v for v in row['outputs'].values()):
                row['F_swap_difference'] = geometry.fundamental_equivalence(*[v['F_in_physical_BA'] for v in row['outputs'].values()])
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            print(json.dumps(dict(pair_id=pair['pair_id'], status={k:v['status'] for k,v in row['outputs'].items()},
                                  valid={k:v['heldout_background']['valid'] for k,v in row['outputs'].items()})), flush=True)
    torch.cuda.synchronize()
    save_new(out/'manifest.json', dict(status='BOUNDED_OLD_VGGT_CONTROL_COMPLETE', pairs=len(rows),
             actual_model_calls_attempted=attempted_calls, actual_model_calls_completed=completed_calls,
             parent_omega_run=dict(path=str(omega_manifest), sha256=sha256(omega_manifest)),
             checkpoint=dict(path=str(checkpoint), sha256=expected), official_source_commit=old_commit,
             model_strict_state_keys=state_keys, no_head_pruning=True, dtype='fp16 aggregator, official fp32 heads',
             setup_seconds=setup_seconds, setup_through_exports_seconds=time.perf_counter()-started,
             CUDA_allocator_peak_bytes=torch.cuda.max_memory_allocated(), CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
             outputs=dict(path=str(out/'pair_results.jsonl'), sha256=sha256(out/'pair_results.jsonl')),
             original_source_hashes={str(p.relative_to(old_source)):sha256(p) for p in old_source.glob('vggt/**/*.py')},
             torch=torch.__version__, torch_cuda=torch.version.cuda,
             no_new_GT=True, no_GMT_inference=True, inference_seed=20261016,
             timing_qualification='old full forward also emits world_points; Omega emits camera/depth only; measured costs not isolated architecture efficiency',
             main_416_scientific_goal_complete=False, provenance=source))


if __name__ == '__main__':
    main()

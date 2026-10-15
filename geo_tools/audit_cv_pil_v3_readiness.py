"""CPU-only old-evidence and producer-metadata preflight; never mines events."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import torch

from cv_pil_dev12_assets import SCHEMA, verify_frozen_evidence
from cv_pil_native_handoff import sha256, safe_torch_load
from cv_pil_v2_model import CandidatePredictor, parameter_digest


def audit(repo, producer, runtime):
    frozen_path = repo/'results/cv_pil_v3_native_validation/FROZEN_PRIOR_EVIDENCE.json'
    frozen = json.loads(frozen_path.read_text())
    preserved = verify_frozen_evidence(frozen)
    old = json.loads((repo/'results/cv_pil_v2_20261015/RESULTS.json').read_text())
    heads = {}
    for arm in ('C2', 'C3', 'C4'):
        t = old['training'][arm]; file = Path(t['local_checkpoint'])
        assert sha256(file) == t['checkpoint_sha256']
        state = safe_torch_load(file)
        assert state['arm'] == arm and state['steps'] == 300
        assert state['seed'] == old['protocol']['seed']
        assert state['cohort_sha256'] == old['protocol']['cohort_sha256']
        model = CandidatePredictor(use_camera=arm != 'C4')
        model.load_state_dict(state['state_dict'], strict=True)
        assert parameter_digest(model) == t['final_parameter_sha256']
        assert sum(p.numel() for p in model.parameters()) == 38080
        heads[arm] = dict(path=str(file), sha256=t['checkpoint_sha256'],
                          parameter_sha256=t['final_parameter_sha256'], parameters=38080,
                          steps=300, seed=state['seed'], unchanged=True)
    # Read only manifest/index metadata. No events, images, labels or tensors
    # from the producer's new TRAIN capture are opened here.
    releases = []
    for path in sorted(runtime.rglob('*.json')):
        if 'sources' in path.relative_to(runtime).parts or path.stat().st_size > 2**20:
            continue
        try:
            item = json.loads(path.read_text())
        except (UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(item, dict) and item.get('schema') == SCHEMA:
            releases.append(dict(path=str(path), sha256=sha256(path), status=item.get('status'),
                                 admitted=False, reason='Requires independent provenance and scope pins'))
    capture = producer/'geo_tools/capture_trackvggt_selected_v3.py'
    plan = producer/'results/trackvggt_v3_native_capture_complete_v1.json'
    return dict(schema='cvpil-v3-readiness/v1', written_utc=datetime.now(timezone.utc).isoformat(),
                frozen_prior_evidence_sha256=sha256(frozen_path),
                preserved_artifacts_and_weights=preserved, heads=heads,
                producer_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=producer).decode().strip(),
                producer_capture_source_sha256=sha256(capture),
                producer_selected_train_completion_sha256=sha256(plan) if plan.is_file() else None,
                candidate_dev12_release_headers=releases,
                status='NEEDS_RELEASE_VALIDATION' if releases else 'BLOCKED_MISSING_NATIVE_ASSETS',
                accepted_dev12_release=None, models_retrained=False, full_video_run=False,
                full_native_OFF_parity='PENDING_REAL_ASSETS_AND_GATED_NATIVE_INTEGRATION',
                synthetic_OFF_boundary_test_is_real_video_parity=False,
                gpu_wall_hours=0, old_ledger_modified=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--producer', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    if a.out.exists(): raise FileExistsError(a.out)
    result = audit(Path(__file__).resolve().parents[1], a.producer, a.runtime)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], preserved=result['preserved_artifacts_and_weights'],
                          old_heads_verified=len(result['heads']), gpu_wall_hours=0)))


if __name__ == '__main__': main()

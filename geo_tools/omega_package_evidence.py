"""Publish small numerical inputs/outputs; never copy images or dense tensors."""
import json
from pathlib import Path
import subprocess
import numpy as np
from omega_common import ROOT, RUNTIME, provenance, save_new, sha256


def main():
    runs = {}
    correspondences = []
    for name in ['omega512_initial24_v1', 'old_vggt_initial24_v1']:
        path = RUNTIME/'pose_audits'/name/'manifest.json'
        manifest = json.loads(path.read_text())
        assert sha256(manifest['outputs']['path']) == manifest['outputs']['sha256']
        rows = [json.loads(line) for line in Path(manifest['outputs']['path']).read_text().splitlines()]
        clock_path = RUNTIME/'logs'/(name+'_receipt.json')
        clock = json.loads(clock_path.read_text())
        assert sha256(clock['log']['path']) == clock['log']['sha256']
        runs[name] = dict(manifest=manifest, manifest_sha256=sha256(path), process_clock=clock, rows=rows)
        if name.startswith('omega512'):
            for row in rows:
                corr = row['correspondence_file']
                assert sha256(corr['path']) == corr['sha256']
                with np.load(corr['path']) as data:
                    correspondences.append(dict(pair_id=row['pair_id'], original_npz_sha256=corr['sha256'],
                                                arrays={k:data[k].tolist() for k in data.files}))
    sizes = {}
    for p in sorted(RUNTIME.iterdir()):
        result = subprocess.check_output(['du', '-sb', str(p)], text=True)
        sizes[p.name] = int(result.split('\t')[0])
    data_bytes = sum(v for k,v in sizes.items() if k not in ['venv', 'weights'])
    assert data_bytes < 2*(1 << 30)
    save_new(ROOT/'results/omega_numeric_evidence_v1.json', dict(
        status='ACTUAL_SMALL_NUMERICAL_EVIDENCE_NO_IMAGES_OR_DENSE_FEATURES', runs=runs,
        putative_correspondences=correspondences,
        old_control_limitation='Old control retains derived F/shape/mapping, not the transient old E/K; Omega E/K are retained. No replay to invent missing old camera records.',
        provenance=provenance()))
    save_new(ROOT/'results/omega_storage_and_setup_v1.json', dict(
        status='ACTUAL_CURRENT_LOGICAL_FOOTPRINT_NOT_LIFETIME_PEAK', root=str(RUNTIME), entries_bytes=sizes,
        additional_nonweight_nonvenv_bytes=data_bytes, research_cache_budget_bytes=2*(1 << 30),
        model_weights_and_environment_separate=True,
        license=dict(path=str(RUNTIME/'weights/LICENSE.txt'), sha256=sha256(RUNTIME/'weights/LICENSE.txt'),
                     git_blob_sha=subprocess.check_output(['git','hash-object',str(RUNTIME/'weights/LICENSE.txt')], text=True).strip(),
                     matches_official_HF_LICENSE_blob='01465aec627c2698f972ca68abcabf54a497db14'),
        environment_freeze=dict(path=str(RUNTIME/'source_audit/isolated_environment_freeze_v1.txt'), sha256=sha256(RUNTIME/'source_audit/isolated_environment_freeze_v1.txt')),
        setup_failure='First isolated pip attempt failed on typing-extensions metadata/build dependency; raw log retained. Second attempt uses modern pip and PyPI plus exact official cu118 Torch wheels. No global or GMT environment modified.',
        setup_logs=[dict(path=str(p),sha256=sha256(p)) for p in sorted((RUNTIME/'logs').glob('install_isolated_*log'))],
        GPU_probe_outer_wall='NOT_CAPTURED', model_producer_costs='complete measured receipts in numeric evidence; not a complete historical GPU bill',
        provenance=provenance()))
    print(json.dumps(dict(status='PACKAGED_NUMERICAL_EVIDENCE', additional_nonweight_nonvenv_bytes=data_bytes)))


if __name__ == '__main__':
    main()

"""Capture local locks/source/credentials-presence without logging private values."""
import json
import os
from pathlib import Path
import subprocess
import sys
from omega_common import BASE, DATA, ROOT, RUNTIME, provenance, save_new, sha256


def main():
    source = RUNTIME / 'official_vggt_omega'
    official = json.loads((RUNTIME / 'source_audit/facebook--VGGT-Omega.json').read_text())
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert source_commit == '48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()
    locked = {
        Path('/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth'): 'cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8',
        ROOT / 'configs/VISION_test.yaml': 'bdaeca71d875e8824c3eaf825967a7ba032514297642a6aabe7ad10a740be94a',
        DATA / 'annotations/train.json': '9093c36204bae482c2f74997f531bfe2d8a46464f4541093703ef22db0068c85',
        Path('/data1/liuyeqiang/WWW_trackvggt_runtime/cv_pil_dev12_native_release_v1/artifact_sha256.json'): 'ef7f65d0bccb1e665f7125391f4a3c64a1449b1fe4d204c735371522386421d4',
    }
    receipts = []
    for path, expected in locked.items():
        actual = sha256(path)
        assert actual == expected, str(path)
        receipts.append(dict(path=str(path), sha256=actual, bytes=path.stat().st_size))
    documents = ['GEO_STREAM_FINAL_REPORT', 'GEO_STREAM_GEOMETRY_FAILURES', 'GEO_STREAM_REPRODUCIBILITY',
                 'TRACKVGGT_FINAL_REPORT', 'TRACKVGGT_FULL_TRAIN_AUDIT', 'TRACKVGGT_CORRECTABLE_ERROR_ANALYSIS',
                 'TRACKVGGT_DEV12_NATIVE_ASSET_HANDOFF', 'TRACKVGGT_REQUIREMENT_AUDIT']
    env_names = ['HF_TOKEN', 'HF_HUB_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN']
    tokens = [Path.home()/'.cache/huggingface/token', Path.home()/'.huggingface/token']
    if os.environ.get('HF_HOME'):
        tokens.append(Path(os.environ['HF_HOME'])/'token')
    if os.environ.get('HF_TOKEN_PATH'):
        tokens.append(Path(os.environ['HF_TOKEN_PATH']))
    branch = subprocess.check_output(['git', 'symbolic-ref', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    assert branch == 'research/vggt-omega-geometry-proof-20261016'
    remote_receipt = RUNTIME / 'logs/reference_remote_HEAD_v1.log'
    remote_line = BASE+'\trefs/heads/research/trackvggt-identity-query-www27-20261015'
    assert remote_line in remote_receipt.read_text().splitlines()
    import torch, numpy, cv2, PIL
    result = dict(schema='omega-preflight/v1', status='LOCAL_LOCKS_AND_OFFICIAL_SOURCE_VERIFIED',
                  scope='metadata/source audit only; no Omega weights loaded and no GMT rerun',
                  parent_commit=BASE, branch=branch, remote_reference_unchanged=True,
                  remote_receipt=dict(path=str(remote_receipt), sha256=sha256(remote_receipt)),
                  locked_inputs=receipts,
                  historical_documents=[dict(path='docs/'+name+'.md', sha256=sha256(ROOT/'docs'/(name+'.md'))) for name in documents],
                  official_source=dict(url='https://github.com/facebookresearch/vggt-omega', commit=source_commit,
                                       path=str(source), files={str(p.relative_to(source)): sha256(p) for p in source.rglob('*')
                                                               if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}),
                  official_HF_revision=official['sha'], official_HF_gated=official['gated'],
                  official_weight_files=[s for s in official['siblings'] if s['rfilename'].endswith('.pt')],
                  weight_access_416='BLOCKED_WEIGHT_ACCESS', official_416_HTTP_status=401,
                  public_512='PUBLIC_REDISTRIBUTION_DOWNLOAD_IN_PROGRESS_ACTUAL_SHA_PENDING',
                  credential_env_presence={name: bool(os.environ.get(name)) for name in env_names},
                  credential_file_presence=[dict(path=str(p), exists=p.is_file()) for p in tokens],
                  credentials_values_logged=False,
                  CPU_preparation_environment=dict(python=sys.version, torch=torch.__version__, torch_cuda=torch.version.cuda,
                     numpy=numpy.__version__, opencv=cv2.__version__, pillow=PIL.__version__),
                  official_minimum_torch='2.6', CPU_environment_meets_official_torch_minimum=False,
                  inference_environment='separate venv, torch2.6.0+cu118/torchvision0.21.0 installation in progress; existing GMT environment unchanged',
                  hardware_receipt=dict(path=str(RUNTIME/'logs/hardware_preflight_v1.log'), sha256=sha256(RUNTIME/'logs/hardware_preflight_v1.log')),
                  real_Omega_inference='NOT_EXECUTED', geometry_success='NOT_MEASURED',
                  scientific_goal_complete=False, provenance=provenance())
    save_new(ROOT/'results/omega_preflight_v1.json', result)
    print(json.dumps({k:result[k] for k in ['status','weight_access_416','public_512','real_Omega_inference']}))


if __name__ == '__main__':
    main()

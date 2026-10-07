"""Phase VI paths, frozen bindings and isolated controller loading."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'reproduction_tools', ROOT / 'third_party/CenterNet2'):
    sys.path.insert(0, str(path))
OUT = Path('/home/liuyeqiang/WWW_jev_phase6_runtime/20261008')
PHASE5 = Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007')
REPORTS = ROOT / 'reports/JEV_PHASE6'
B2 = PHASE5 / 'minimal_training/B2/calibration/model_calibrated.pth'
B2_SHA = 'f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7'
PYTHON = '/home/liuyeqiang/anaconda3/envs/GMT/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    from run_jev_phase5_ablation import save as atomic_save
    atomic_save(path, value)


def protect_anchor():
    if sha(B2) != B2_SHA:
        raise RuntimeError('the permanent B2 positive anchor changed')


def new_output(path):
    path = Path(path).resolve()
    if OUT not in path.parents or path.exists():
        raise RuntimeError(f'refusing reused or non-Phase-VI output: {path}')
    path.mkdir(parents=True)
    return path


def load_controller(path, device='cpu'):
    import torch
    from gtr.modeling.jev_lifecycle_gates import MatchThresholdGate
    from gtr.modeling.jev_runtime import build_controller_from_checkpoint, TemperatureScaledController
    payload = torch.load(str(path), map_location='cpu')
    name = payload.get('model_name', 'jev')
    if name not in {'phase6_scalar_gate', 'phase6_dynamic_gate'}:
        return build_controller_from_checkpoint(path, device=device)
    model = MatchThresholdGate(payload['state_dim'], payload['hidden_dim']).to(device)
    model.load_state_dict(payload['model'], strict=True)
    temperature = float(payload.get('calibration_temperature', 1.0))
    return TemperatureScaledController(model, temperature).to(device).eval()

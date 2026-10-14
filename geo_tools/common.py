"""Shared provenance and pure-GMT loading; never imports JEV."""
import hashlib, json, os, random, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('GEO_RUNTIME', '/data1/liuyeqiang/WWW_vggt_geo_runtime'))
DATA = Path('/data/DATASETS/TRACKING/JDE/VisionTrack')
CHECKPOINT = Path('/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth')
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/CenterNet2')]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)

def seed_all(seed):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

def load_gmt(seed=20261014, geo_opts=None):
    import torch, gtr
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from centernet.config import add_centernet_config
    from gtr.config import add_gtr_config
    seed_all(seed)
    cfg = get_cfg(); add_centernet_config(cfg); add_gtr_config(cfg)
    cfg.merge_from_file(str(ROOT / 'configs/VISION_test.yaml'))
    cfg.MODEL.WEIGHTS = str(CHECKPOINT)
    if geo_opts:
        for key, value in geo_opts.items(): setattr(cfg.MODEL.GEO, key, value)
    model = build_model(cfg)
    state = torch.load(CHECKPOINT, map_location='cpu')['model']
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return cfg, model

def train_data():
    return json.loads((DATA / 'annotations/train.json').read_text())

def video_records(data, video_id, max_frames=None):
    rows = sorted([x for x in data['images'] if x['video_id'] == video_id], key=lambda x: (x['view_id'], x['frame_id']))
    views = sorted({x['view_id'] for x in rows})
    if max_frames is not None: rows = [x for x in rows if x['frame_id'] <= max_frames]
    keys = [{x['frame_id'] for x in rows if x['view_id'] == v} for v in views]
    if any(k != keys[0] for k in keys): raise ValueError('not synchronized; explicit timestamp alignment required')
    return [dict(x, image_id=x['id'], file_name=str(DATA / 'train' / x['file_name']), view_num=len(views)) for x in rows]

def map_records(cfg, records):
    from gtr.data.gtr_dataset_mapper import GMTDatasetMapper
    from gtr.data.custom_build_augmentation import build_custom_augmentation
    mapper = GMTDatasetMapper(cfg, False, augmentations=build_custom_augmentation(cfg, False))
    return mapper({'images': records, 'video_id': records[0]['video_id'], 'view_num': records[0]['view_num']})

def provenance(seed=20261014):
    import torch, numpy
    return dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                command=sys.argv, seed=seed, python=sys.version, torch=torch.__version__, numpy=numpy.__version__,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), used_cache=False,
                checkpoint=str(CHECKPOINT), checkpoint_sha256=sha256(CHECKPOINT),
                config_sha256=sha256(ROOT/'configs/VISION_test.yaml'),
                annotation_sha256=sha256(DATA/'annotations/train.json'))

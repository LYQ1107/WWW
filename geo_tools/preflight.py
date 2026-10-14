import json, subprocess
from common import *

def main():
    data = train_data()
    families = ['garden', 'square', 'football', 'park', 'basketball']
    dev = [v['id'] for v in data['videos'] if any(v['file_name'][5:] == f for f in families)]
    split = dict(version=1, seed=20261014, source='original TRAIN only', unit='complete scene family',
                 association_train_video_ids=[v['id'] for v in data['videos'] if v['id'] not in dev],
                 full_development_video_ids=dev, minimal_development_video_ids=[1,14,22],
                 backbone_exposure='GMT checkpoint trained on original TRAIN; development is NOT backbone-unseen',
                 historical_test_exposure=True, official_test_used_for_this_project=False)
    save_json(ROOT/'manifests/geo_split_v1.json', split)
    cfg, model = load_gmt()
    manifest = provenance(); manifest.update(status='CHECKPOINT_STRICT_LOAD_PASS_BASELINE_PENDING',
        official_source_origin='8fa96705 + missing source files from dfa9ca8e + mapper compatibility from cbae5f9a',
        gmt_state_keys=len(model.state_dict()), split=split,
        hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu', '--format=csv'], text=True),
        protocol='original offline GMT with MIN_TRACK_LEN=50 and full-video input; not causal online',
        dataset_source='existing VisionTrack installation; historical download records to audit',
        jev_policy_or_weights_or_cache_reused=False)
    save_json(ROOT/'results/baseline_manifest.json', manifest)
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__': main()

"""Immutable inventory for the pre-registered Ω-FullTrain 24-scene study.

This is a read-only CPU audit. It does not run GMT, read image pixels, alter
raw annotations, or select a model. It records every TRAIN/TEST sequence,
camera, frame count, GT hash, duplicate/conflict frame and existing causal
audit provenance so that later training cannot silently use a partial scope.
"""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/data/DATASETS/TRACKING/JDE/VisionTrack')
PROTOCOL = ROOT / 'manifests/omega_full24_training_protocol_v1.json'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def gt_scan(path):
    rows = 0
    valid = 0
    malformed = 0
    duplicate_pairs = collections.Counter()
    pair_frames = collections.defaultdict(set)
    identities = collections.defaultdict(set)
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        rows += 1
        try:
            values = [float(value) for value in line.split(',')]
            if len(values) < 8:
                raise ValueError('short row')
            frame, identity = int(values[0]), int(values[1])
        except (ValueError, OverflowError):
            malformed += 1
            continue
        pair = (frame, identity)
        duplicate_pairs[pair] += 1
        pair_frames[pair].add(line_no)
        identities[frame].add(identity)
        if values[4] > 0 and values[5] > 0 and frame >= 1 and identity >= 0 and values[6] > 0 and values[7] == 1:
            valid += 1
    duplicate_pair_rows = sum(count - 1 for count in duplicate_pairs.values() if count > 1)
    duplicate_frames = sorted({frame for frame, identity in duplicate_pairs if duplicate_pairs[(frame, identity)] > 1})
    # A frame with the same (frame,identity) pair more than once is the raw
    # identity conflict class used by the prior GT audit.  Do not compare all
    # identities with a per-frame list (which would make this scan quadratic).
    conflict_frames = duplicate_frames[:]
    return dict(rows=rows, valid_rows=valid, malformed_rows=malformed,
                duplicate_identity_pair_count=sum(count > 1 for count in duplicate_pairs.values()),
                duplicate_identity_rows=duplicate_pair_rows, duplicate_frames=duplicate_frames,
                conflict_frames=conflict_frames, frame_count=len(identities),
                identity_count=len({identity for values in identities.values() for identity in values}))


def annotation_inventory(split):
    annotation = DATA / 'annotations' / f'{split}.json'
    payload = json.loads(annotation.read_text())
    by_video = collections.defaultdict(list)
    for image in payload['images']:
        by_video[int(image['video_id'])].append(image)
    videos = {}
    for video in payload['videos']:
        video_id = int(video['id'])
        images = by_video[video_id]
        views = sorted({int(image['view_id']) for image in images})
        frames = sorted({int(image['frame_id']) for image in images})
        cameras = {}
        for view in views:
            rows = [image for image in images if int(image['view_id']) == view]
            frame_ids = sorted({int(image['frame_id']) for image in rows})
            cameras[str(view)] = dict(images=len(rows), first_frame=frame_ids[0], last_frame=frame_ids[-1], frames=len(frame_ids))
            if frame_ids != frames:
                raise RuntimeError(f'{split} video {video_id} camera {view} is not synchronized')
        name = video['file_name']
        gt = {}
        root = DATA / split
        for view in views:
            path = root / f'{name}_View{view}' / 'gt' / 'gt.txt'
            if not path.is_file():
                raise FileNotFoundError(path)
            gt[str(view)] = dict(path=str(path), sha256=sha256(path), **gt_scan(path))
        videos[str(video_id)] = dict(name=name, images=len(images), views=views, frames=len(frames), first_frame=frames[0], last_frame=frames[-1], cameras=cameras, gt=gt)
    return dict(annotation_path=str(annotation), annotation_sha256=sha256(annotation), video_count=len(payload['videos']), image_count=len(payload['images']), videos=videos)


def existing_audits():
    root = Path('/data1/liuyeqiang/WWW_trackvggt_runtime/full_train_audit_v1')
    result = {}
    for manifest in sorted(root.glob('scene*_full_v1/manifest.json')):
        value = json.loads(manifest.read_text())
        video_id = int(value['video_id'])
        artifacts = {path.name: sha256(path) for path in manifest.parent.iterdir() if path.is_file()}
        result[str(video_id)] = dict(path=str(manifest.parent), manifest_sha256=sha256(manifest), status=value.get('status'), full_video=value.get('full_video'), groups=value.get('groups'), images=value.get('images'), primary_queries=value.get('primary_calls'), artifacts=artifacts)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    train = annotation_inventory('train')
    test = annotation_inventory('test')
    train_ids = sorted(map(int, train['videos']))
    expected = protocol['dataset']['video_ids']
    if train_ids != expected:
        raise RuntimeError(f'protocol TRAIN ids {expected} != annotation ids {train_ids}')
    if train['annotation_sha256'] != protocol['dataset']['annotation_sha256_required']:
        raise RuntimeError('TRAIN annotation hash changed')
    fit = protocol['scene_split']['fit_video_ids']
    validation = protocol['scene_split']['validation_video_ids']
    if sorted(fit + validation) != expected or set(fit) & set(validation):
        raise RuntimeError('locked scene split is not an exact partition')
    report = dict(status='PASS_IMMUTABLE_FULL24_TRAIN_TEST_INVENTORY', protocol_path=str(PROTOCOL), protocol_sha256=sha256(PROTOCOL), train=train, test=test, existing_full_causal_audits=existing_audits(), split=dict(fit=fit, validation=validation, all24=expected), raw_files_mutated=False, source_worktree=str(ROOT), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), command=sys.argv, generated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=report['status'], train_images=train['image_count'], test_images=test['image_count'], train_scenes=len(train['videos']), test_scenes=len(test['videos']), existing_full_audits=sorted(map(int, report['existing_full_causal_audits'])))))


if __name__ == '__main__':
    main()

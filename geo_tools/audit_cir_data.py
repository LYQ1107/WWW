"""Read-only complete TRAIN inventory and TEST availability, never edits GT."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root, split):
    annotation = root / 'annotations' / (split + '.json')
    data = json.loads(annotation.read_text())
    images = defaultdict(list)
    labels = defaultdict(list)
    for row in data['images']:
        images[row['video_id']].append(row)
    for row in data.get('annotations', []):
        labels[row['image_id']].append(row)
    videos = []
    for video in data['videos']:
        rows = images[video['id']]
        cameras = defaultdict(list)
        missing = []
        zero_bytes = []
        for row in rows:
            cameras[row['view_id']].append(row['frame_id'])
            path = root / split / row['file_name']
            if not path.is_file():
                missing.append(str(path))
            elif path.stat().st_size == 0:
                zero_bytes.append(str(path))
        persons = set()
        repeated_events = 0
        raw_gt = []
        for camera in sorted(cameras):
            dirname = Path(next(r['file_name'] for r in rows if r['view_id'] == camera)).parent.parent
            path = root / split / dirname / 'gt/gt.txt'
            counts = Counter()
            if path.is_file():
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    fields = line.split(',')
                    frame, person = map(lambda x: int(float(x)), fields[:2])
                    persons.add(person)
                    counts[(frame, person)] += 1
                repeated_events += sum(n > 1 for n in counts.values())
            raw_gt.append(dict(camera=camera, path=str(path), available=path.is_file(),
                               sha256=sha(path) if path.is_file() else None,
                               duplicate_frame_identity_keys=sum(n > 1 for n in counts.values())))
        frame_lists = [sorted(f) for f in cameras.values()]
        videos.append(dict(video_id=video['id'], name=video['file_name'], images=len(rows),
            cameras={str(k):dict(images=len(v), first=min(v), last=max(v),
                unique_frames=len(set(v)), continuous=sorted(v)==list(range(min(v),max(v)+1))) for k,v in cameras.items()},
            synchronized=all(f == frame_lists[0] for f in frame_lists),
            missing_images=missing[:20], missing_image_count=len(missing), zero_byte_images=zero_bytes[:20],
            raw_gt=raw_gt, raw_identity_count=len(persons), duplicate_gt_event_count=repeated_events,
            annotation_rows=sum(len(labels[r['id']]) for r in rows)))
    return dict(split=split, annotation=str(annotation), annotation_sha256=sha(annotation),
                videos=videos, total_images=len(data['images']), total_videos=len(videos),
                missing_image_count=sum(v['missing_image_count'] for v in videos),
                image_check='stat every declared image; no decoder/content hash scan yet',
                test_identity_semantics_used=False if split=='test' else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('/data/DATASETS/TRACKING/JDE/VisionTrack'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    # TEST availability uses metadata and file existence only, no identity labels.
    train = inventory(a.data, 'train')
    test_annotation = a.data/'annotations/test.json'
    test = dict(annotation_available=test_annotation.is_file(), image_root_available=(a.data/'test').is_dir(),
                annotation_sha256=sha(test_annotation) if test_annotation.is_file() else None,
                inference_executed=False, GT_semantics_read=False,
                historical_TEST_exposure=True, exposure_source='inherited geo_split_v1 and GMT_CURRENT_DIAGNOSTIC_REPORT_20261005; not an untouched benchmark')
    if test_annotation.is_file():
        data=json.loads(test_annotation.read_text())
        test.update(videos=len(data['videos']), images=len(data['images']),
                    missing_images=sum(not (a.data/'test'/r['file_name']).is_file() for r in data['images']))
    result=dict(status='INVENTORY_COMPLETE_NOT_TRAINING_ADMISSION', TRAIN=train, TEST_availability=test,
                qualification='duplicate raw GT event labels will be masked; no GT modification or whole-video deletion',
                native_cache_coverage='pending exact per-scene source SHA audit; prior 11 assets are not full24',
                source_sha256=sha(__file__))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],videos=train['total_videos'],images=train['total_images'],missing=train['missing_image_count'],TEST=test)))


if __name__ == '__main__':
    main()

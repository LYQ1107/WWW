#!/usr/bin/env python3
"""Read-only split/GT audit and inference-only metadata export."""
import argparse, collections, hashlib, json
from pathlib import Path


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(root, out):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in ('train', 'test'):
        source = root / 'annotations' / f'{split}.json'
        data = json.loads(source.read_text())
        images = {int(x['id']): x for x in data['images']}
        counts = collections.Counter((int(a['image_id']), int(a['instance_id'])) for a in data['annotations'])
        ambiguous = [list(k) for k, n in counts.items() if n > 1]
        ann_rows = collections.defaultdict(collections.Counter)
        for ann in data['annotations']:
            im = images[int(ann['image_id'])]
            ann_rows[(int(im['video_id']), int(im['view_id']))][
                (int(im['frame_id']), int(ann['instance_id']), *map(float, ann['bbox']))] += 1
        meta = {k: data[k] for k in ('videos', 'images', 'categories') if k in data}
        metapath = out / f'{split}_metadata.json'
        metapath.write_text(json.dumps(meta, separators=(',', ':')))
        scenes = []
        for video in data['videos']:
            sid, name = int(video['id']), video['file_name']
            records = [im for im in images.values() if int(im['video_id']) == sid]
            frames = {}
            for im in records: frames.setdefault(int(im['view_id']), set()).add(int(im['frame_id']))
            common = sorted(set.intersection(*frames.values()))
            raw = []
            for view in sorted(frames):
                p = root / split / f'{name}_View{view}' / 'gt/gt.txt'
                keys = collections.Counter()
                box_rows = collections.Counter()
                n = 0
                if p.exists():
                    for line in p.read_text().splitlines():
                        vals = line.strip().split(',')
                        if len(vals) >= 6:
                            keys[(int(float(vals[0])), int(float(vals[1])))] += 1; n += 1
                            box_rows[(int(float(vals[0])), int(float(vals[1])), *map(float, vals[2:6]))] += 1
                raw.append({'view': view, 'path': str(p), 'exists': p.exists(), 'sha256': digest(p) if p.exists() else None,
                            'rows': n, 'duplicate_keys': sum(v > 1 for v in keys.values()),
                            'duplicate_rows_extra': sum(v-1 for v in keys.values()),
                            'max_frame': max((k[0] for k in keys), default=0),
                            'same_annotation_rows': box_rows == ann_rows[(sid, view)],
                            'frame_indices_inside_sequence': all(k[0] in frames[view] for k in keys),
                            'same_max_frame_as_metadata': max((k[0] for k in keys), default=0) == max(frames[view], default=0)})
            scenes.append({'scene': sid, 'name': name, 'synchronized_frames': len(common), 'first_frame': common[0] if common else None,
                           'last_frame': common[-1] if common else None, 'images': len(records), 'raw_gt': raw,
                           'official_gt_legal': all(x['exists'] and x['duplicate_keys'] == 0 and x['frame_indices_inside_sequence'] and x['same_annotation_rows'] for x in raw),
                           'sample_images_exist': all((root / split / x['file_name']).exists() for x in records[::max(1,len(records)//8)])})
        r = {'split': split, 'annotation_sha256': digest(source), 'scenes': scenes, 'scene_count': len(scenes),
             'synchronized_timestamps': sum(x['synchronized_frames'] for x in scenes), 'images': len(images),
             'annotations': len(data['annotations']), 'ambiguous_annotation_keys': len(ambiguous),
             'ambiguous_annotation_keys_sha256': hashlib.sha256(json.dumps(ambiguous, sort_keys=True).encode()).hexdigest(),
             'raw_gt_modified': False, 'metadata_only': str(metapath), 'metadata_sha256': digest(metapath)}
        (out / f'{split}_ambiguities.json').write_text(json.dumps({'ambiguous_keys': ambiguous}))
        results[split] = r
        print(split, r['scene_count'], r['synchronized_timestamps'], r['annotations'], 'ambiguous', len(ambiguous), 'legal', sum(x['official_gt_legal'] for x in scenes), flush=True)
    (out / 'data_audit.json').write_text(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset-root', default='/data/DATASETS/TRACKING/JDE/VisionTrack'); p.add_argument('--output', required=True)
    a=p.parse_args(); audit(a.dataset_root, a.output)

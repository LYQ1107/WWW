"""CPU-only original TRAIN census; reuse the hash-verified GT uniqueness audit."""
import collections
import csv
import io
import numpy as np
from common import ROOT, DATA, Path, json, sha256, save_json, video_records, train_data

NEW = Path('/data1/liuyeqiang/WWW_trackvggt_runtime')


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, min=None, p25=None, median=None, p75=None, p95=None, max=None, mean=None)
    q = np.quantile(a, [0, .25, .5, .75, .95, 1])
    return dict(n=len(a), **dict(zip(['min', 'p25', 'median', 'p75', 'p95', 'max'], map(float, q))), mean=float(a.mean()))


def patch_count(box, width, height):
    # Same half-open center convention and round-to-14 pad mapping as PadMapping.
    w, h = (518, round(height * 518 / width / 14) * 14) if width >= height else (round(width * 518 / height / 14) * 14, 518)
    x, y, bw, bh = box
    x1, y1 = x*w/width+(518-w)//2, y*h/height+(518-h)//2
    x2, y2 = (x+bw)*w/width+(518-w)//2, (y+bh)*h/height+(518-h)//2
    centers = (np.arange(37)+.5)*14
    content_x = (centers >= (518-w)//2) & (centers < (518-w)//2+w)
    content_y = (centers >= (518-h)//2) & (centers < (518-h)//2+h)
    return int(((centers >= x1) & (centers < x2) & content_x).sum() * ((centers >= y1) & (centers < y2) & content_y).sum())


def patch_bin(count):
    return '0' if count == 0 else '1-3' if count < 4 else '4-15' if count < 16 else '16+'


def verify_v2():
    cohort_path = ROOT/'results/trackvggt_expanded_cohort_v2.json'
    c = json.loads(cohort_path.read_text())
    sources = dict(c['source_artifacts'])
    for name, digest in sources.items():
        if sha256(name) != digest:
            raise ValueError('locked V2 artifact changed: '+name)
    token_manifests = {}; total = 0
    for backbone in ['vggt', 'dinov2']:
        keys = set()
        for scene in [4, 5, 6, 7, 8, 12]:
            p = NEW/'tokens'/f'{backbone}_scene{scene}_full_v2'/'manifest.json'
            m = json.loads(p.read_text())
            if m['status'] != 'COMPLETE_BOUNDED_REAL_TARGET_TOKEN_POOL':
                raise ValueError('unfinished V2 features')
            for row in m['rows']:
                if tuple(row['key']) in keys or sha256(row['file']) != row['sha256']:
                    raise ValueError('duplicate/changed token')
                keys.add(tuple(row['key'])); total += Path(row['file']).stat().st_size
            token_manifests[str(p)] = sha256(p)
        if keys != {tuple(r['key']) for split in ['fit', 'validation'] for r in c[split]}:
            raise ValueError('V2 feature membership changed')
    queues = {}
    for p in sorted((NEW/'queues').glob('trackvggt_phase2_tokens_v2_gpu*/manifest.json')):
        m = json.loads(p.read_text())
        if m['status'] != 'FINISHED' or not all(r['status'] == 'PASS' for r in m['stages']):
            raise ValueError('unfinished V2 extraction queue')
        queues[str(p)] = sha256(p)
    return dict(status='VERIFIED_IMMUTABLE_V2', cohort_sha256=sha256(cohort_path), fit=len(c['fit']), validation=len(c['validation']),
                raw_source_artifacts=len(sources), token_manifests=token_manifests, token_bytes=total, queues=queues,
                selected_reachable_errors=c['fit_reachable_known_errors'], selected_known_ranking_errors=c['fit_known_ranking_errors'])


def main():
    data = train_data(); split = json.loads((ROOT/'manifests/geo_split_v1.json').read_text())
    quality_path = ROOT/'results/train_gt_quality_audit.json'; quality = json.loads(quality_path.read_text())
    trusted = {(r['video_id'], r['view_id']):r for r in quality['rows']}
    rows = []; details = []; hashes = {}; missing_files = []
    for video in sorted(data['videos'], key=lambda v:v['id']):
        scene = video['id']; images = video_records(data, scene); meta = {(i['view_id'], i['frame_id']):i for i in images}
        cameras = sorted({i['view_id'] for i in images}); frames = sorted({i['frame_id'] for i in images})
        identity_frames = collections.defaultdict(set); identity_cameras = collections.defaultdict(set)
        by_frame = collections.defaultdict(list); by_view_frame = collections.Counter(); instances = []; patches = collections.Counter()
        view_rows = []; invalid = 0; outside = 0; excluded = 0
        for camera in cameras:
            p = DATA/'train'/f"{video['file_name']}_View{camera}"/'gt/gt.txt'
            if not p.exists():
                missing_files.append(str(p)); raise FileNotFoundError(p)
            digest = sha256(p); previous = trusted[scene, camera]
            if digest != previous['raw_sha256']:
                raise ValueError('GT changed; uniqueness reuse unsafe: '+str(p))
            hashes[str(p)] = digest; raw_rows = p.read_text().splitlines()
            if len(raw_rows) != previous['raw_rows']:
                raise ValueError('GT row census changed')
            for line in raw_rows:
                v = np.asarray(list(map(float, line.split(','))))
                if len(v) < 8 or not np.isfinite(v).all() or v[4] <= 0 or v[5] <= 0 or v[0] < 1 or v[1] < 0:
                    invalid += 1; continue
                frame, identity = int(v[0]), int(v[1]); image = meta.get((camera, frame))
                if image is None:
                    outside += 1; continue
                if v[6] <= 0 or v[7] != 1:
                    excluded += 1; continue
                box = v[2:6]; count = patch_count(box, image['width'], image['height'])
                patches[patch_bin(count)] += 1
                instances.append([box[2], box[3], box[2]*box[3], count])
                by_frame[frame].append((identity, camera)); by_view_frame[camera, frame] += 1
                identity_frames[identity].add(frame); identity_cameras[identity].add(camera)
            view_rows.append(dict(camera=camera, frames=sum(i['view_id']==camera for i in images), **previous))
        trajectories = [dict(identity=k, first=min(v), last=max(v), span_groups=max(v)-min(v)+1, visible_groups=len(v), cameras=sorted(identity_cameras[k])) for k,v in sorted(identity_frames.items())]
        cross = [sum(len({camera for identity,camera in by_frame[f] if identity == ident}) > 1 for ident in {i for i,c in by_frame[f]}) for f in frames]
        simultaneous = [len(by_frame[f]) for f in frames]
        targets = np.asarray(instances).reshape(-1,4)
        single = sum(len(v)==1 for v in identity_cameras.values()); multi = len(identity_cameras)-single
        role = 'association_validation' if scene == 12 else 'association_fit' if scene in split['association_train_video_ids'] else 'strict_DEV5' if scene in split['strict_development_video_ids'] else 'conflicting_GT_DEV_diagnostic'
        row = dict(video_id=scene, scene=video['file_name'], role=role, status='COMPLETE_CPU_CENSUS', cameras=len(cameras),
                   camera_frames={v:sum(i['view_id']==v for i in images) for v in cameras}, synchronized_groups=len(frames),
                   gt_raw_rows=sum(r['raw_rows'] for r in view_rows), gt_valid_target_rows=len(instances), unique_raw_gt_identities=len(identity_frames),
                   simultaneous_target_rows=distribution(simultaneous), cross_camera_simultaneous_identities=distribution(cross), cross_camera_identity_group_occurrences=sum(cross),
                   identity_span_groups=distribution([r['span_groups'] for r in trajectories]), identity_visible_groups=distribution([r['visible_groups'] for r in trajectories]),
                   single_camera_identities=single, multi_camera_identities=multi, multi_camera_identity_fraction=multi/max(1,single+multi),
                   bbox_width_px=distribution(targets[:,0]), bbox_height_px=distribution(targets[:,1]), bbox_area_px=distribution(targets[:,2]),
                   patch_0=patches['0'], patch_1_3=patches['1-3'], patch_4_15=patches['4-15'], patch_16_plus=patches['16+'],
                   duplicate_identity_extra_rows=sum(r['duplicate_identity_rows'] for r in view_rows), conflicting_bbox_groups=sum(r['conflicting_bbox_groups'] for r in view_rows),
                   exact_duplicate_groups=sum(r['exact_duplicate_groups'] for r in view_rows), invalid_rows=invalid, gt_rows_outside_image_timestamps=outside,
                   non_target_or_inactive_rows=excluded, missing_gt_files=0, missing_label_annotations='NOT_MEASURED',
                   zero_target_image_frames=sum(by_view_frame[c,f]==0 for c in cameras for f in frames))
        rows.append(row); details.append(dict(**row, per_identity_trajectory=trajectories, views=view_rows))
        print(json.dumps(dict(scene=scene, groups=len(frames), targets=len(instances), identities=len(identity_frames), conflicts=row['conflicting_bbox_groups'], patch_bins=dict(patches))), flush=True)
    if {r['video_id'] for r in rows} != set(range(1,25)):
        raise ValueError('all24 original TRAIN required')
    output = io.StringIO(); writer = csv.DictWriter(output,fieldnames=list(rows[0])); writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    (ROOT/'results/trackvggt_all24_dataset_census_v1.csv').write_text(output.getvalue())
    v2 = verify_v2()
    summary = dict(status='COMPLETE_ALL24_CPU_CENSUS', source='original TRAIN only; TEST never opened',
                   prior_uniqueness_audit=str(quality_path), prior_uniqueness_audit_sha256=sha256(quality_path), uniqueness_reused_exact_hashes=True,
                   annotation_sha256=sha256(DATA/'annotations/train.json'), raw_gt_sha256=hashes, v2_lock=v2, rows=details,
                   count_policy='Target instances are original valid active class1 rows, including duplicates: no GT repair. Identity and cross-camera counts use raw IDs and are descriptive, not identity correctness claims in conflicted sequences.',
                   patch_policy='PadMapping518, patch14, half-open centers inside valid resized content; patch count is not visual information or ROIAlign usability.',
                   missing_policy='No GT file is missing. Empty labeled image frames do not establish missing annotation; unannotated objects cannot be counted from GT alone.')
    save_json(ROOT/'results/trackvggt_gt_quality_summary_v1.json',summary)
    lines = ['# VisionTrack original TRAIN dataset difficulty census', '',
             'CPU-only census of all24 original TRAIN sequences. All48 raw GT hashes match the existing uniqueness audit, whose duplicate/conflict counts are reused. No VGGT, tracking replay, TEST data or GT repair. Original valid active class1 rows, including duplicates, are counted; raw identities in conflicting sequences remain descriptive.', '',
             '|Scene|Role|Groups|GT rows|Raw identities|Multi-camera identities|Patch0 /1–3 /4–15 /16+|Conflicting groups|', '|---|---|---:|---:|---:|---:|---|---:|']
    for r in rows:
        lines.append(f"|{r['video_id']} {r['scene']}|{r['role']}|{r['synchronized_groups']}|{r['gt_valid_target_rows']}|{r['unique_raw_gt_identities']}|{r['multi_camera_identities']}|{r['patch_0']} /{r['patch_1_3']} /{r['patch_4_15']} /{r['patch_16_plus']}|{r['conflicting_bbox_groups']}|")
    lines += ['', 'Patch coverage uses the actual518 pad mapping and14-pixel centers. A zero-center box may still support ROIAlign. GT size statistics cannot establish detector failures, appearance discriminability, or candidate headroom; those require the causal audit. Missing annotation count is NOT_MEASURED. Empty labeled frames are reported separately.', '',
              f"Locked Cohort-V2 verification: {v2['fit']} fit /{v2['validation']} validation, all raw artifacts and2600 token files hash-verified; {v2['token_bytes']} bytes. Four extraction queues finished with all stages PASS. No teacher training has started."]
    (ROOT/'docs/TRACKVGGT_DATASET_DIFFICULTY_AUDIT.md').write_text('\n'.join(lines)+'\n')
    plan = dict(id='trackvggt_full_audit_v1', status='CPU_CENSUS_COMPLETE_CAUSAL_AUDIT_PENDING', reference_commit='0c4a3a94a9a51ea8f6d05b9756b70133182ebfdc',
                request_sha256=sha256(ROOT/'docs/TRACKVGGT_URGENT_AUDIT_ADDENDUM.md'), project_gpu_ids=['0','4','5','6'], inference_seed=20261014,
                priority_tiers=[[9,13,17,20,21,24],[4,5,6,7,8],[12]], scene_plan=[dict(scene=r['video_id'], groups=r['synchronized_groups'], images=2*r['synchronized_groups'], role=r['role'], causal_status='INCOMPLETE', raw_gt_conflicts=r['conflicting_bbox_groups']) for r in rows if r['video_id'] in split['association_train_video_ids']],
                continuous_from_first_group=True, old_prefix400_comparison_required=[4,5,6,7,8,12], immutable_v2=v2,
                max_retained_events_per_scene=1000, full_audit_output=str(NEW/'full_train_audit_v1'), no_foundation_encoder_during_scan=True,
                gt_conflict_policy='Preserve all raw rows. Mark every camera-frame containing a duplicate (frame,identity) or invalid GT unreliable for proposal labeling, no sanitized official metrics. Report resulting unknown coverage.',
                labeling='After full inference and independent OFF prefix parity. Stream compact ranks/source indices; no persistent full-history feature vectors or dense score matrices.',
                dev5='Reuse saved full outputs/metrics only; missing actual solver/candidate evidence NOT_MEASURED. No GT-driven fit/model selection.',
                validation='Original300 unchanged; scene12 full audit is separate retrospective diagnostic, never fit selection.',
                v3='Only after full inventory. If justified, bounded2000–4000 unique scene-rotating events, unchanged validation300. A later selected-event capture may need continuous GMT replay because the audit intentionally retains no native feature archive; this is feature construction, not a second statistical scan.',
                teacher_training_priority='Prepared AG code is unexecuted; pause launch pending this prerequisite audit.')
    save_json(ROOT/'manifests/trackvggt_full_audit_plan_v1.json',plan)


if __name__ == '__main__':
    main()

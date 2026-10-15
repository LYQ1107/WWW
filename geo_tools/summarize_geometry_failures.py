"""Observed gate failures, not inferred physical/identity failure causes."""
import collections
import csv
import math
from common import *


def value(row,key):
    try:result=float(row[key])
    except (KeyError,ValueError,TypeError):return None
    return result if math.isfinite(result) else None


def classify(family,row):
    violations=[];unknown=[]
    def lower(key,bound,label):
        number=value(row,key)
        if number is None:unknown.append(key)
        elif number<bound:violations.append(label)
    if family in {'traditional_F','plane_H'}:
        lower('inliers',20,'fit_inliers_below20')
        lower('inlier_ratio',.25,'logged_fit_ratio_below0.25')
        lower('coverage',.01,'logged_footprint_below1percent')
        median=value(row,'median_residual')
        if median is None:violations.append('required_residual_statistic_unavailable')
        elif median>3:violations.append('logged_median_above3_original_px')
    else:
        key='median_epipolar_original_px' if family=='vggt_pose' else 'median_residual'
        median=value(row,key)
        lower('coverage',.01,'logged_footprint_below1percent')
        if family=='vggt_pose':
            lower('background_matches',20,'tentative_background_matches_below20')
            unknown.append('residual_point_count_not_separately_logged')
        else:
            lower('candidate_inliers',20,'neural_candidate_inliers_below20')
            lower('validation_matches',20,'independent_reference_points_below20')
            unknown.append('independent_pixel_inlier_count_not_separately_logged')
        if median is None:
            violations.append('required_residual_statistic_unavailable')
            # Zero ratios are also early-return sentinels in these source
            # routines. Do not call them measured bad residuals when absent.
            unknown.append('residual_ratio_unavailable_after_early_return')
        else:
            lower('inlier_ratio_at_12px' if family=='vggt_pose' else 'inlier_ratio',.25,'measured_ratio_below0.25_at12px')
            if median>12:violations.append('measured_median_above12_original_px')
        unknown.append('matrix_shape_finiteness_rank_not_logged')
    valid=row['valid']=='True'
    if valid and violations:raise ValueError('recorded acceptance contradicts observed gate values')
    if not valid and not violations:unknown.append('rejection_not_explained_by_logged_fields')
    return violations,unknown


def source_snapshot(info,file):
    context=info.get('provenance',{});declared=context.get('commit',info.get('source_commit'))
    captured=context.get('source_hashes',{}).get(file);resolved=None;code=None
    candidates=[declared] if declared else []
    if captured:candidates+=subprocess.check_output(['git','log','--all','--format=%H','--',file],cwd=ROOT,text=True).splitlines()
    for commit in dict.fromkeys(candidates):
        try:current=subprocess.check_output(['git','show',commit+':'+file],cwd=ROOT,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:continue
        if captured is None or hashlib.sha256(current).hexdigest()==captured:
            resolved=commit;code=current;break
    if code is None:raise ValueError('source rule snapshot unavailable: '+file)
    return dict(file=file,declared_commit=declared,captured_file_sha256=captured,matching_git_commit=resolved,
        resolved_file_sha256=hashlib.sha256(code).hexdigest(),original_tree_dirty=context.get('source_tree_dirty'),
        verification='CAPTURED_FILE_BYTES_MATCH_GIT_BLOB' if captured else 'DECLARED_GIT_SOURCE_ONLY_NO_CAPTURED_FILE_HASH')


def main():
    profiles=[('traditional_F','geometry_validation.csv','geometry_reference_manifest.json','gtr/geometry/geometry_confidence.py'),
        ('plane_H','plane_geometry_validation.csv','plane_geometry_manifest.json','gtr/geometry/geometry_confidence.py'),
        ('vggt_pose','vggt_geometry_validation.csv','vggt_geometry_manifest.json','geo_tools/validate_vggt_geometry.py'),
        ('neural_F_H','neural_background_sym_train_dev_v1.csv','neural_background_sym_train_dev_v1_manifest.json','gtr/geometry/background_patch_matcher.py')]
    split=json.loads((ROOT/'manifests/geo_split_v1.json').read_text());rows=[];groups=collections.defaultdict(list);artifacts={};snapshots={}
    for family,csv_name,manifest_name,source in profiles:
        path=ROOT/'results'/csv_name;manifest=ROOT/'results'/manifest_name;info=json.loads(manifest.read_text())
        if 'csv_sha256' in info and info['csv_sha256']!=sha256(path):raise ValueError('changed diagnostic CSV')
        artifacts.update({str(p):sha256(p) for p in (path,manifest)});snapshots[family]=source_snapshot(info,source)
        for original in csv.DictReader(path.open()):
            scene=int(original['video_id']);cohort='association_train' if scene in split['association_train_video_ids'] else 'strict_dev' if scene in split['strict_development_video_ids'] else 'other_recorded_scene'
            violations,unknown=classify(family,original)
            row=dict(family=family,cohort=cohort,video_id=scene,frame_id=int(original['frame_id']),
                method=original.get('method',original.get('feature','')+'/'+original.get('geometry','')),
                masking=original.get('masked','foreground/background masks fixed by neural/plane profile'),
                recorded_valid=original['valid']=='True',violations=violations,unlogged_or_unresolved=unknown)
            rows.append(row);groups[(family,cohort)].append(row)
    summaries=[]
    for (family,cohort),values in sorted(groups.items()):
        summaries.append(dict(family=family,cohort=cohort,diagnostic_rows=len(values),valid=sum(row['recorded_valid'] for row in values),
            video_ids=sorted({row['video_id'] for row in values}),
            unique_scene_frames=len({(row['video_id'],row['frame_id']) for row in values}),
            gate_violation_counts=dict(collections.Counter(reason for row in values for reason in row['violations'])),
            unlogged_or_unresolved_counts=dict(collections.Counter(reason for row in values for reason in row['unlogged_or_unresolved']))))
    dense=[]
    spec=ROOT/'manifests/causal_B7_fixed_k1_strict_suite_v1_sources.json'
    for scene,prediction in json.loads(spec.read_text())['video_predictions'].items():
        path=Path(prediction).parent/'manifest.json';run=json.loads(path.read_text())
        if sha256(prediction)!=run['predictions_sha256']:raise ValueError('changed dense predictions')
        dense.append(dict(video_id=int(scene),synchronized_groups=run['synchronized_groups'],
            numerical_geometry_valid=run['geometry_valid_calls'],numerical_geometry_invalid=run['geometry_invalid_calls'],
            module_failures=run['geo_failures'],actual_vggt_calls=run['total_vggt_calls'],manifest_sha256=sha256(path)))
    text=['# Geometry recovery: recorded gate diagnostics','',
        'These are logged gate values on fixed screening cases. Repeated mask/feature/geometry variants are not independent samples. GT identity labels and downstream scores do not choose these gates. Threshold violations are not evidence of low physical camera overlap, low texture, rapid rotation or GT pose error. Missing matrix/residual-stage details remain unknown.','',
        '| Profile | Cohort | Rows | Unique scene/frames | Accepted |', '|---|---|---:|---:|---:|']
    for row in summaries:text.append(f"| {row['family']} | {row['cohort']} | {row['diagnostic_rows']} | {row['unique_scene_frames']} | {row['valid']} |")
    text+=['','Traditional F/plane H: at least20 fit inliers, ratio≥.25, finite median≤3 original pixels, footprint≥1%. VGGT pose: at least20 finite independent background residuals, ratio≥.25 within12 original pixels, median≤12, footprint≥1%. Neural candidates additionally require20 fit inliers and independent pixel inlier support; missing support/matrix fields cannot be reconstructed as a physical cause.','',
        'Mask labels differ by profile: traditional F masks keypoints, VGGT masks the model input while its pixel reference remains background-masked, and neural matching excludes detector footprints and padding. Homography is a soft plane approximation and does not establish arbitrary 3D propagation.','',
        f"The separate complete primary dense causal run records{sum(row['numerical_geometry_valid'] for row in dense)} numerical/background gate passes, at{sum(row['actual_vggt_calls'] for row in dense)} actual VGGT calls. Its full B5/B7 output attribution is independently audited; screening rejection counts must not replace full-run validity counts or GT pose accuracy.",'',
        'Some early diagnostics captured dirty source trees. Matching a recorded file SHA to a later committed Git blob recovers those exact file bytes; it does not make the original whole tree clean. The earliest traditional reference only records a source commit, so that weaker provenance is explicit. JSON contains source snapshots, per-row violations and unlogged fields.','']
    (ROOT/'docs/GEO_STREAM_GEOMETRY_FAILURES.md').write_text('\n'.join(text))
    save_json(ROOT/'results/geometry_failure_taxonomy_v1.json',dict(status='RECORDED_GATE_DIAGNOSTICS_CONSOLIDATED',summaries=summaries,rows=rows,
        source_artifacts=artifacts,source_rule_snapshots=snapshots,dense_primary_separate_scope=dense,gt_used_for_gate_classification=False,
        limitations='descriptive diagnostic gate accounting, not error-cause attribution, GT pose validation, full-DEV failure reasons, tracker throughput or general geometry impossibility',provenance=provenance()))
    print(json.dumps(dict(diagnostic_rows=len(rows),accepted=sum(row['recorded_valid'] for row in rows),dense_numerical_gate_passes=sum(row['numerical_geometry_valid'] for row in dense))))


if __name__=='__main__':main()

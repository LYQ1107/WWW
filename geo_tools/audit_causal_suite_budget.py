"""Audit realized compute and whole-run timing of complete causal suites."""
import argparse
from common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sources',type=Path,nargs='+',required=True);parser.add_argument('--tag',required=True);args=parser.parse_args()
    strict=sorted(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    methods=[]
    for source in args.sources:
        spec=json.loads(source.read_text());predictions={int(video):Path(path) for video,path in spec['video_predictions'].items()}
        if sorted(predictions)!=strict:raise ValueError('all complete strict scenes required')
        rows=[]
        for video,path in sorted(predictions.items()):
            manifest=path.parent/'manifest.json';run=json.loads(manifest.read_text())
            if not run['causal_online'] or not run['full_video'] or run['video_id']!=video:raise ValueError('complete causal protocol required')
            if sha256(path)!=run['predictions_sha256']:raise ValueError('changed prediction file')
            schedule=Path(run['schedule_trace'])
            if sha256(schedule)!=run['schedule_trace_sha256']:raise ValueError('changed schedule trace')
            trace=json.loads(schedule.read_text())
            chosen=sum(row['chosen'] for row in trace)
            cap=run['geo_options']['TOTAL_CALL_BUDGET']
            actual=run['total_vggt_calls']
            if cap>=0 and actual>cap:raise ValueError('actual encoder calls exceed total cap')
            rows.append(dict(video_id=video,manifest=str(manifest),manifest_sha256=sha256(manifest),commit=run['commit'],
                images=run['images'],groups=run['synchronized_groups'],seconds=run['seconds'],actual_vggt_calls=actual,actual_dinov2_calls=run['dinov2_calls'],
                selected_updates=chosen,reported_updates=run['geo_calls'],declared_total_cap=cap,valid_geometry_calls=run['geometry_valid_calls'],
                invalid_geometry_calls=run['geometry_invalid_calls'],module_failures=run['geo_failures'],peak_gpu_bytes=run['peak_gpu_bytes'],peak_reserved_gpu_bytes=run['peak_reserved_gpu_bytes'],
                input_to_identity_p50=run['input_to_identity_p50'],input_to_identity_p95=run['input_to_identity_p95']))
        seconds=sum(row['seconds'] for row in rows);images=sum(row['images'] for row in rows);groups=sum(row['groups'] for row in rows)
        methods.append(dict(source=str(source),source_sha256=sha256(source),rows=rows,total_images=images,total_groups=groups,
            sum_processing_seconds=seconds,cumulative_camera_image_fps=images/seconds,cumulative_group_fps=groups/seconds,
            total_actual_vggt_calls=sum(row['actual_vggt_calls'] for row in rows),total_actual_dinov2_calls=sum(row['actual_dinov2_calls'] for row in rows),
            maximum_process_gpu_peak=max(row['peak_gpu_bytes'] for row in rows),total_valid_geometry=sum(row['valid_geometry_calls'] for row in rows),
            total_module_failures=sum(row['module_failures'] for row in rows)))
    matched_by_scene=all(len({method['rows'][index]['actual_vggt_calls'] for method in methods})==1 for index in range(len(strict)))
    save_json(ROOT/'results'/f'{args.tag}.json',dict(status='PASS',methods=methods,matched_actual_vggt_calls_for_each_scene=matched_by_scene,
        scope='all strict complete causal videos; input/read/detector/GTA/geometry/CPU identity serialization charged by each original run',
        timing_scope='sum of measured processing seconds; not calendar wall time including waits between independent experiments; static GMT setup excluded, per-video cold lazy VGGT included',
        limitations='single descriptive full-suite measurements on V100; no repeated timing or equal-accuracy Pareto superiority claim; per-video P50/P95 cannot be averaged into a global latency percentile',provenance=provenance()))


if __name__=='__main__':main()

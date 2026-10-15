"""Freeze complete continuous TRAIN12 assets, without consulting GT/errors."""
import collections
import shutil
from common import *
from dev12_native_schema import INTERFACE


def main():
    target=ROOT/'manifests/trackvggt_dev12_native_asset_release_v1.json'
    if target.exists():raise FileExistsError(target)
    images=video_records(train_data(),12);groups=collections.defaultdict(list)
    for image in images:groups[image['frame_id']].append(image)
    if sorted(groups)!=list(range(1,1201)) or any(len(v)!=2 for v in groups.values()):raise ValueError('unexpected full scope')
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')
    if shutil.disk_usage(runtime).free<40*(1<<30):raise ValueError('insufficient dedicated output disk')
    prov=provenance()
    model_sources={k:v for k,v in SOURCE_HASHES.items() if k.startswith(('gtr/modeling/','gtr/geometry/','gtr/data/','gtr/evaluation/')) or k in ('gtr/config.py','geo_tools/common.py','geo_tools/dev12_native_schema.py')}
    model_sources.update({str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'configs').rglob('*.yaml')})
    value=dict(interface_version=INTERFACE,status='FROZEN_GT_INDEPENDENT_SCOPE',video_id=12,range_inclusive=[1,1200],
        physical_camera_order=[r['view_id'] for r in groups[1]],physical_input_image_ids=[r['image_id'] for rows in groups.values() for r in rows],
        input_image_sha256={str(r['image_id']):sha256(r['file_name']) for r in images},
        query_retention='Every actual native primary query, including empty-candidate and unknown; largest-camera initial ID births recorded separately as commits, never invented as association queries',
        frozen_provenance={k:prov[k] for k in ['checkpoint','checkpoint_sha256','config_sha256','annotation_sha256','seed']},
        bound_source_hashes=model_sources,capture_source_sha256=sha256(ROOT/'geo_tools/export_gmt_dev12_native_assets.py'),
        source_commit_before_scope_creation=SOURCE_COMMIT,allowed_gpu_ids=[0,4,5,6],execution_gpu_id=5,
        maximum_total_gpu_wall_hours=4.0,maximum_single_process_seconds=10800,maximum_output_bytes=4*(1<<30),minimum_free_bytes=40*(1<<30),
        target_directory=str(runtime/'cv_pil_dev12_native_release_v1'),precheck_directory=str(runtime/'dev12_native_precheck_v1'),
        data_precision='Native FP32 1152 dimensions, no conversion, no normalized identity-only replacement',
        labels='Separate offline_labels after all native inference and parity gates. No GT labels/features used for scope/candidates/history.',
        disclosure='Expanded development assets under a GT-independent selection rule. TRAIN12 labels have previously been viewed; this is not an unseen blind test.',
        resource_ledger='Independent dev12_native_v1 ledger includes prechecks, failed attempts and full capture; no old resource ledger reset',
        do_not_modify=['V2 original1000/300','V3 fixed3000','seeds','GMT core/weights/config','V3 queues','sealed full audit','other branches/weights'],
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save_json(target,value);print(json.dumps(dict(scope=str(target),sha256=sha256(target),images=len(images),groups=len(groups))))


if __name__=='__main__':main()

"""CPU repacking of a frozen real native prefix, not fresh image inference."""
import argparse
import json
from pathlib import Path
import resource
import time
from cv_pil_dev12_shards import NativeShardAssets
from pirga_indexed_store import StoreWriter, StoreReader, sparse_matrix, sha


def pack(assets,out,scene,pins):
    writer=StoreWriter(out);features={}
    for key,o in assets.observations.items():
        features[(key,None)]=writer.add_feature(scene,list(key),o['frame'],o['camera'],o['feature'].numpy())
    for vid,v in assets.variants.items():
        key=tuple(v['source']);o=assets.observations[key]
        features[(key,vid)]=writer.add_feature(scene,list(key),o['frame'],o['camera'],v['feature'].numpy(),variant=vid)
    def fid(spec):return features[(tuple(spec['source']),spec['variant'])]
    for _,c in sorted(assets.calls.items()):
        shape,entries=sparse_matrix(c['track_scores'].numpy())
        writer.add_call(dict(scene=scene,call_id=c['call_id'],stage=c['stage'],query_keys=[[scene,*q] for q in c['queries']],
            query_features=[fid(s) for s in c['query_feature_keys']],reference_features=[fid(s) for s in c['reference_feature_keys']],
            reference_tracks=[r['track'] for r in c['references']],candidate_ids=c['track_ids'],reference_block_sizes=c['reference_block_sizes'],
            matrix_shape=shape,matrix_entries=entries,thresholds=c['thresholds'].tolist(),accepted_ids=c['decision']['accepted_track_ids'],
            completed_image_ids=c['completed_camera_images_before_call']))
    return writer.close(pins,len(assets.query_index))


def main():
    p=argparse.ArgumentParser();p.add_argument('--pins',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();pins=json.loads(a.pins.read_text());started=time.monotonic()
    import torch
    torch.set_num_threads(1)
    assets=NativeShardAssets(a.root/'manifest.json',pins['upstream_manifest_sha256'],pins['scope_sha256'],pins['provenance_pins'],pins['roles'],prefix_engineering_only=True,progress=False)
    read=time.monotonic();m=pack(assets,a.out,12,dict(prefix_manifest=pins['upstream_manifest_sha256'],independent_pin_file_sha256=sha(a.pins)))
    packed=time.monotonic();reader=StoreReader(a.out,sha(a.out/'manifest.json'));calls=list(reader.calls())
    if len(calls)!=len(assets.calls):raise ValueError('Packed call lost')
    for c in calls:
        orig=assets.calls[c['call_id']]
        if c['accepted_ids']!=orig['decision']['accepted_track_ids'] or c['query_keys']!=[[12,*s] for s in orig['queries']]:raise ValueError('Original actions changed')
    r=dict(status='PASS_REAL_PREFIX_INDEXED_REPACK_ENGINEERING_ONLY',scene=12,groups=40,queries=len(assets.query_index),calls=len(calls),features=m['features'],
           input_read_validation_seconds=read-started,packing_seconds=packed-read,roundtrip_seconds=time.monotonic()-packed,
           output_bytes=sum(p.stat().st_size for p in a.out.rglob('*') if p.is_file()),quantization=m['quantization'],
           peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GT_read=False,training=False,gpu_wall_hours=0,
           fresh_image_export=False,fresh_export_wall_seconds=None,original_primary_actions_exact=True,full11_export_budget_established=False,
           limitation='Sealed native DEV12 prefix repacked on CPU. Does not replace a fresh fit-scene native export resource precheck or certify FP16-feature GTA equality.')
    (a.out/'PROBE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

if __name__=='__main__':main()

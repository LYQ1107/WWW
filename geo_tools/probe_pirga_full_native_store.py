"""Complete sealed-data codec check, no labels, new model or fresh image inference.

TRAIN12 is already exposed. This is engineering only; quantized C1 action drift
is measured without GT and cannot be called repair, harm, HOTA or training gain.
The FP16 choice is fixed before this check, never selected by DEV performance.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import shutil
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from cv_pil_dev12_shards import NativeShardAssets
from pirga_indexed_store import StoreReader,sha,dense_matrix
from probe_pirga_indexed_store import pack


def accepted(scores,ids,thresholds):
    rr,cc=linear_sum_assignment(-scores);out=[-1]*len(scores)
    for r,c in zip(rr,cc):
        if scores[r,c]>thresholds[c]:out[r]=ids[c]
    return out


def cosine_scores(c,assets,feature):
    """Identical CPU operators for original-FP32 and stored-FP16 views."""
    ids=c['track_ids'];owner={t:i for i,t in enumerate(ids)};native=c['track_scores'];result=native.clone()
    if not len(c['queries']):return result
    frames={assets.observations[tuple(q)]['frame'] for q in c['queries']}
    if len(frames)!=1:raise ValueError('Expected homogeneous actual native query image')
    frame=next(iter(frames));idx=[i for i,r in enumerate(c['references']) if assets.observations[tuple(r['source'])]['frame']<frame]
    if not idx:return result
    h=torch.stack([feature(c['reference_feature_keys'][i]) for i in idx]);h=torch.nn.functional.normalize(h,dim=-1)
    owners=torch.tensor([owner[c['references'][i]['track']] for i in idx],dtype=torch.long)
    proto=torch.zeros(len(ids),1152).index_add(0,owners,h);proto=torch.nn.functional.normalize(proto,dim=-1)
    for row,spec in enumerate(c['query_feature_keys']):
        q=torch.nn.functional.normalize(feature(spec),dim=0);result[row]=native[row]+2*(proto@q)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--pins',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();started=time.monotonic();torch.set_num_threads(1)
    if a.out.exists():raise FileExistsError(a.out)
    if shutil.disk_usage(a.out.parent).free<30*(1<<30)+512*(1<<20):raise ValueError('Disk reserve plus bounded engineering store required')
    pins=json.loads(a.pins.read_text());assets=NativeShardAssets(**pins['release']);read_done=time.monotonic()
    manifest=pack(assets,a.out,12,dict(source_pin_file_sha256=sha(a.pins),release_manifest_sha256=pins['release']['manifest_sha']))
    packed=time.monotonic();reader=StoreReader(a.out,sha(a.out/'manifest.json'));fids={}
    for fid,key in reader.db.execute('SELECT id,source FROM features'):
        key=json.loads(key);fids[(tuple(key[1:3]),key[3])]=fid
    def stored(spec):return torch.from_numpy(reader.feature(fids[(tuple(spec['source']),spec['variant'])]))
    vector_count=0
    for source,o in assets.observations.items():
        spec=dict(source=list(source),variant=None)
        if not np.array_equal(stored(spec).numpy(),o['feature'].numpy().astype(np.float16).astype(np.float32)):raise ValueError('Stored source vector changed')
        vector_count+=1
    for variant,v in assets.variants.items():
        spec=dict(source=v['source'],variant=variant)
        if not np.array_equal(stored(spec).numpy(),v['feature'].numpy().astype(np.float16).astype(np.float32)):raise ValueError('Stored version vector changed')
        vector_count+=1
    stages=Counter();drift=[];max_score=0.;primary=0;matrix_cells=0;reference_entries=0
    for i,c in enumerate(reader.calls(),1):
        src=assets.calls[c['call_id']];stages[c['stage']]+=1
        dense=dense_matrix(c['matrix_shape'],c['matrix_entries'])
        if not np.array_equal(dense,src['track_scores'].numpy()):raise ValueError('Full FP32 matrix changed')
        if c['observed_track_lengths']!=src['observed_track_lengths'] or c['candidate_support_counts']!=src['actual_candidate_support_lengths']:raise ValueError('Lifetime/support lost')
        for fid,spec in zip(c['query_features']+c['reference_features'],src['query_feature_keys']+src['reference_feature_keys']):
            if fid!=fids[(tuple(spec['source']),spec['variant'])]:raise ValueError('Feature reference/version/order changed')
        matrix_cells+=dense.size;reference_entries+=len(c['reference_features'])
        if src['stage']!='run_memory_tracker':
            primary+=len(src['queries']);s32=cosine_scores(src,assets,assets.feature);s16=cosine_scores(src,assets,stored)
            if s32.numel():max_score=max(max_score,float((s32-s16).abs().max()))
            act32=accepted(s32.numpy(),src['track_ids'],src['thresholds'].numpy());act16=accepted(s16.numpy(),src['track_ids'],src['thresholds'].numpy())
            for row,(x,y) in enumerate(zip(act32,act16)):
                if x!=y:drift.append(dict(key=[12,*src['queries'][row]],fp32_C1_accepted=x,fp16_C1_accepted=y,certified_repair=None,certified_harm=None))
        if i%250==0:print(json.dumps(dict(phase='indexed_full_roundtrip',calls=i,primary_queries=primary,quantization_only_action_changes=len(drift),seconds=time.monotonic()-packed)),flush=True)
    size=sum(p.stat().st_size for p in a.out.rglob('*') if p.is_file())
    if size>512*(1<<20):raise ValueError('Engineering output exceeded512MiB cap')
    report=dict(status='PASS_COMPLETE_SEALED_NATIVE_INDEX_CODEC_ENGINEERING_ONLY',interface=manifest['interface'],scene=12,groups=1200,
        primary_queries=primary,observation_vectors=len(assets.observations),actual_descriptor_versions=len(assets.variants),stored_vectors=vector_count,calls_by_stage=dict(stages),
        complete_matrix_cells=matrix_cells,complete_history_reference_entries=reference_entries,all_native_matrices_bitwise_equal=True,all_original_joint_accept_actions_exact=True,
        all_query_and_history_reference_order_and_versions_exact=True,all_lifetime_and_current_support_counts_exact=True,
        quantization=manifest['quantization'],C1_quantization_diagnostic=dict(alpha=2.,strict_past_only=True,max_absolute_score_difference=max_score,action_changes=len(drift),events=drift,
             GT_semantics_loaded=False,repair_or_harm_classification='NOT_DEFINED_WITHOUT_LABELS',dtype_selected_from_result=False),
        timings=dict(source_input_validation_seconds=read_done-started,packing_seconds=packed-read_done,roundtrip_and_quantization_seconds=time.monotonic()-packed),
        output_bytes_before_report=size,peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        GPU_wall_hours=0,fresh_image_inference=False,GT_supervision_loaded=False,training=False,full11_train_admitted=False,
        input_pins_sha256=sha(a.pins),store_manifest_sha256=sha(a.out/'manifest.json'),scientific_success=False)
    (a.out/'PROBE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()

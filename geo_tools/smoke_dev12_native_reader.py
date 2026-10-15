"""Actual sealed-reader traversal of every query, with GT/video/GMT absent."""
import argparse
import json
import time
from pathlib import Path
import torch
from read_dev12_native_release import NativeRelease
from validate_dev12_native_inputs import file_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--seal-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    start=time.perf_counter();data=NativeRelease(a.release,a.seal_sha256,with_offline_labels=False)
    queries=0;candidate_histories=0;history_references=0
    for key in data.queries:
        item=data.query(key)
        if 'offline_label' in item or data.labels is not None:raise ValueError('GT read by default consumer')
        if item['current_feature'].shape!=(1152,) or item['current_feature'].dtype!=torch.float32:raise ValueError('current vector interface')
        if [r['track_id'] for r in item['histories']]!=item['call']['track_ids']:raise ValueError('candidate order changed')
        if item['commit']['track']!=item['final_track']:raise ValueError('final commit interface')
        for history in item['histories']:
            if history['features'].shape!=(history['actual_context_length'],1152):raise ValueError('missing history vectors')
            candidate_histories+=1;history_references+=history['actual_context_length']
        queries+=1
    if queries!=data.manifest['native_queries']:raise ValueError('native query completeness')
    result=dict(status='PASS_ALL_QUERY_SEALED_READER',queries=queries,candidate_histories_materialized=candidate_histories,
        history_reference_vectors_materialized=history_references,seal_sha256=a.seal_sha256,manifest_sha256=file_sha(a.release/'manifest.json'),
        source_sha256=file_sha(Path(__file__).with_name('read_dev12_native_release.py')),every_final_commit_linked=True,
        all_artifact_SHAs_verified=True,GT_labels_loaded=False,images_read=False,GMT_inference_calls=0,seconds=time.perf_counter()-start)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

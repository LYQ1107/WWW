"""Independent read-only consumer; no GMT, video, or GT imports/reads."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from dev12_native_schema import INTERFACE, activate, digest_json, replay_call, tensor_digest, validate_past


def rows(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:yield json.loads(line)


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()


def validate(root, verify_artifacts=True):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['interface_version']!=INTERFACE:raise ValueError('unsupported schema')
    if verify_artifacts:
        for name,digest in manifest['artifacts'].items():
            if file_sha(root/name)!=digest:raise ValueError('artifact changed '+name)
    observations={}
    for path in sorted((root/'native_observations').glob('observations_*.pt')):
        for key,value in torch.load(path,map_location='cpu').items():
            if key in observations:raise ValueError('duplicate physical observation')
            if list(key)!=value['source'] or value['feature'].shape!=(1152,) or value['feature'].dtype!=torch.float32:raise ValueError('native observation contract')
            if tensor_digest(value['feature'])!=value['feature_sha256']:raise ValueError('observation precision changed')
            observations[key]=value
    variants=torch.load(root/'native_observations/descriptor_variants.pt',map_location='cpu')
    def vector(key):
        if key['variant'] is None:return observations[tuple(key['source'])]['feature']
        value=variants[key['variant']]
        if value['source']!=key['source'] or tensor_digest(value['feature'])!=value['feature_sha256']:raise ValueError('descriptor version source mismatch')
        return value['feature']
    queries={};calls={};max_activation_error=0.;max_aggregation_error=0.;bank_calls=0
    for path in sorted((root/'native_calls').glob('calls_*.pt')):
        for call in torch.load(path,map_location='cpu'):
            cid=call['call_id']
            if cid in calls:raise ValueError('duplicate native call')
            calls[cid]=call;memory=call['stage']=='run_memory_tracker';bank_calls+=memory
            q=call['queries'];refs=call['references'];tracks=call['track_ids']
            if tracks!=sorted({r['track'] for r in refs}):raise ValueError('candidate order changed')
            if len(refs)!=len(call['reference_feature_keys']) or len(q)!=len(call['query_feature_keys']):raise ValueError('missing input vector mapping')
            if sum(call['reference_block_sizes'])!=len(refs):raise ValueError('reference frame blocks changed')
            probs=activate(call['raw_logits'],call['activation_block_sizes']);saved=call['single_reference_probabilities']
            if probs.shape!=saved.shape or not torch.allclose(probs,saved,atol=2e-6,rtol=2e-6):raise ValueError('per-reference GTA cannot be restored')
            if probs.numel():max_activation_error=max(max_activation_error,float((probs-saved).abs().max()))
            ids=torch.tensor([r['track'] for r in refs],dtype=torch.long);membership=(ids[:,None]==torch.tensor(tracks,dtype=torch.long)[None,:]).float()
            scores=saved@membership
            if not torch.allclose(scores,call['track_scores'],atol=2e-5,rtol=2e-6):raise ValueError('native aggregation mismatch')
            if scores.numel():max_aggregation_error=max(max_aggregation_error,float((scores-call['track_scores']).abs().max()))
            pairs,accepted=replay_call(call)
            if pairs!=call['decision']['hungarian_pairs'] or accepted!=call['decision']['accepted_track_ids']:raise ValueError('actual joint assignment not reproduced')
            support=[sum(r['track']==track for r in refs) for track in tracks]
            if support!=call['actual_candidate_support_lengths']:raise ValueError('history support length mismatch')
            for source,key in zip(q,call['query_feature_keys']):
                if source!=key['source']:raise ValueError('query source mismatch')
                vector(key)
            for ref,key in zip(refs,call['reference_feature_keys']):
                if ref['source']!=key['source']:raise ValueError('history source mismatch')
                vector(key)
                for source in q:validate_past(observations[tuple(source)],observations[tuple(ref['source'])],set(call['completed_camera_images_before_call']))
            if memory:
                for track,spec in call['bank_descriptor_support'].items():
                    if spec is None:raise ValueError('missing actual Bank averaging provenance')
                    total=0
                    for key in spec['sources']:
                        total=total+vector(key)
                        for source in q:validate_past(observations[tuple(source)],observations[tuple(key['source'])],set(call['completed_camera_images_before_call']))
                    total=total/spec['denominator']
                    for ref,key in zip(refs,call['reference_feature_keys']):
                        if ref['track']==int(track) and not torch.allclose(vector(key),total,atol=2e-6,rtol=2e-6):raise ValueError('actual Bank descriptor support mismatch')
            else:
                for row,source in enumerate(q):
                    key=(12,*source)
                    if key in queries:raise ValueError('duplicate primary physical query')
                    queries[key]=(cid,row)
    query_rows=list(rows(root/'native_queries.jsonl.gz'))
    if len(query_rows)!=len(queries) or len({tuple(r['key']) for r in query_rows})!=len(query_rows):raise ValueError('query stream duplicated/incomplete')
    for row in query_rows:
        key=tuple(row['key']);cid,qi=queries[key];call=calls[cid]
        if (cid,qi)!=(row['call_id'],row['query_row']):raise ValueError('query row linkage')
        if row['final_track']!=call['decision']['final_track_ids'][qi] or row['primary_accepted_track']!=call['decision']['accepted_track_ids'][qi]:raise ValueError('final native submission linkage')
    previous=None;commits=0;last_state_by_call={};commit_keys={};running_counts={}
    for row in rows(root/'native_commit_trace.jsonl.gz'):
        if row['index']!=commits or row['previous_commit_sha256']!=previous:raise ValueError('commit chain changed')
        if digest_json(row['state'])!=row['state_sha256']:raise ValueError('actual state fingerprint')
        value=dict(row);current=value.pop('commit_sha256')
        if digest_json(value)!=current:raise ValueError('commit record fingerprint')
        if row['state']['reid_digests'][str(row['track'])]!=row['reid_digest']:raise ValueError('commit descriptor state mismatch')
        running_counts[str(row['track'])]=running_counts.get(str(row['track']),0)+1
        if running_counts!=row['state']['counts']:raise ValueError('actual per-commit support-count update cannot be reconstructed')
        key=tuple(row['source'])
        if key in commit_keys:raise ValueError('duplicate physical commit')
        commit_keys[key]=row['track'];previous=current;commits+=1
        if row['call_id'] is not None:last_state_by_call[row['call_id']]=row['state']
    for call in calls.values():
        if call['stage']=='run_memory_tracker':continue
        if call['queries'] and last_state_by_call[call['call_id']]!=call['state_after']:raise ValueError('last commit/native method state differs')
        for q,track in zip(call['queries'],call['decision']['final_track_ids']):
            if commit_keys[tuple(q)]!=track:raise ValueError('commit trace/final track mismatch')
    if set(commit_keys)!=set(observations):raise ValueError('detector observations/commit coverage mismatch')
    if manifest['native_queries']!=len(queries) or manifest['total_calls']!=len(calls):raise ValueError('manifest coverage mismatch')
    return dict(status='PASS_READ_ONLY_NATIVE_CONSUMER',interface_version=INTERFACE,queries=len(queries),observations=len(observations),calls=len(calls),
        bank_calls=bank_calls,commits=commits,descriptor_versions=len(variants),native_dtype='torch.float32',lossy_compression=False,
        every_joint_Hungarian_exact=True,every_threshold_acceptance_exact=True,all_query_and_full_reference_features_present=True,
        every_final_commit_linked=True,commit_state_chain_verified=True,GT_read=False,images_read=False,GMT_inference_calls=0,
        cpu_activation_max_abs_error=max_activation_error,cpu_aggregation_max_abs_error=max_aggregation_error,
        numerical_note='CPU softmax/matmul verified with disclosed tolerance; the actual FP32 CPU Hungarian input is saved exactly and yields exact assignments.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=validate(a.release);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

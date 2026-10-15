"""Read-only CV-PIL handoff API. No GMT, images, annotation, or GT dependency.

NativeRelease(..., expected_seal_sha256=published_sha).query((12,image,row))
returns all actual candidate histories and the complete joint assignment call.
Labels are an explicit, separate opt-in; they never select the input rows.
"""
import json
from pathlib import Path
import torch
from validate_dev12_native_inputs import file_sha, rows
from dev12_native_schema import INTERFACE


class NativeRelease:
    def __init__(self, directory, expected_seal_sha256, with_offline_labels=False):
        self.directory=Path(directory);root=self.directory;seal=root/'artifact_sha256.json'
        if file_sha(seal)!=expected_seal_sha256:raise ValueError('published release SHA differs')
        index=json.loads(seal.read_text())
        if index['interface_version']!=INTERFACE or index['status']!='SEALED_ASSET_READY':raise ValueError('unsupported/unsealed release')
        for name,digest in index['artifacts'].items():
            if file_sha(root/name)!=digest:raise ValueError('sealed artifact changed: '+name)
        self.manifest=json.loads((root/'manifest.json').read_text())
        if self.manifest['status']!='SEALED_ASSET_READY':raise ValueError('release not ready')
        self.observations={};self.calls={};self.commits={};self.labels=None
        for path in sorted((root/'native_observations').glob('observations_*.pt')):self.observations.update(torch.load(path,map_location='cpu'))
        self.variants=torch.load(root/'native_observations/descriptor_variants.pt',map_location='cpu')
        for path in sorted((root/'native_calls').glob('calls_*.pt')):
            for call in torch.load(path,map_location='cpu'):self.calls[call['call_id']]=call
        self.queries={tuple(r['key']):r for r in rows(root/'native_queries.jsonl.gz')}
        self.commits={tuple(r['source']):r for r in rows(root/'native_commit_trace.jsonl.gz')}
        if with_offline_labels:self.labels={tuple(r['key']):r for r in rows(root/'offline_labels/queries.jsonl.gz')}

    def vector(self, key):
        value=self.observations[tuple(key['source'])] if key['variant'] is None else self.variants[key['variant']]
        return value['feature']

    def query(self, key):
        key=tuple(key);query=self.queries[key];call=self.calls[query['call_id']];row=query['query_row']
        observation=self.observations[key[1:]];histories=[]
        for track in call['track_ids']:
            indices=[i for i,ref in enumerate(call['references']) if ref['track']==track]
            sources=[call['reference_feature_keys'][i] for i in indices]
            histories.append(dict(track_id=track,reference_columns=indices,feature_keys=sources,
                features=torch.stack([self.vector(s) for s in sources]),
                frames=[self.observations[tuple(s['source'])]['frame'] for s in sources],
                cameras=[self.observations[tuple(s['source'])]['camera'] for s in sources],
                actual_context_length=len(indices),observed_total_length=call['observed_track_lengths'][str(track)]))
        value=dict(key=list(key),current=observation,current_feature=self.vector(call['query_feature_keys'][row]),
            histories=histories,call=call,query_row=row,native_scores=call['track_scores'][row],
            primary_accepted_track=query['primary_accepted_track'],final_track=query['final_track'],commit=self.commits[key[1:]])
        if self.labels is not None:value['offline_label']=self.labels[key]
        return value

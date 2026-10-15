"""Separate observable model inputs from post-run identity supervision."""
import collections,functools
import numpy as np
import torch
from common import *

NEW=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')


class TrackEventData:
    def __init__(self,backbone,cohort_path=None):
        self.backbone=backbone;self.cohort_path=Path(cohort_path or NEW/'cohorts/hard_events_expanded_v2/cohort.json')
        self.cohort=json.loads(self.cohort_path.read_text())
        if self.cohort['status']!='LOCKED_SCENE_DISJOINT_REAL_EVENT_COHORT':raise ValueError('unlocked cohort')
        self.items=self.cohort['fit']+self.cohort['validation'];self.fit=list(range(len(self.cohort['fit'])));self.validation=list(range(len(self.cohort['fit']),len(self.items)))
        self.payloads={};self.rows={};self.source_artifacts={str(self.cohort_path):sha256(self.cohort_path)}
        for path,digest in self.cohort['source_artifacts'].items():
            if sha256(path)!=digest:raise ValueError('locked raw event source changed: '+path)
            self.source_artifacts[path]=digest
        for scene in self.cohort['fit_scene_ids']+self.cohort['validation_scene_ids']:
            path=NEW/'tokens'/f'{backbone}_scene{scene}_full_v2'/'manifest.json';manifest=json.loads(path.read_text())
            if manifest['status']!='COMPLETE_BOUNDED_REAL_TARGET_TOKEN_POOL' or manifest['limited_probe'] or manifest['scene']!=scene:raise ValueError('full target pool missing')
            if manifest['backbone']!=backbone or not manifest['frozen'] or not manifest['no_dense_maps_or_full_attention_saved']:raise ValueError('invalid frozen feature scope')
            self.source_artifacts[str(path)]=sha256(path)
            for row in manifest['rows']:
                key=tuple(row['key'])
                if key in self.rows:raise ValueError('duplicate physical feature query')
                if sha256(row['file'])!=row['sha256']:raise ValueError('target features changed')
                self.rows[key]=row;self.source_artifacts[row['file']]=row['sha256']
        if set(self.rows)!={tuple(item['key']) for item in self.items}:raise ValueError('teacher event memberships differ from locked cohort')
        for path in sorted({item['event_file'] for item in self.items}):
            self.payloads[path]=torch.load(path,map_location='cpu');self.source_artifacts[path]=sha256(path)
        self.native_dim=self.cohort['native_feature_dim'];self.channels=2048 if backbone=='vggt' else 1024

    @functools.lru_cache(maxsize=128)
    def arrays(self,key):
        row=self.rows[key]
        with np.load(row['file']) as data:return {name:data[name].copy() for name in data.files}

    def event(self,index):
        item=self.items[index];payload=self.payloads[item['event_file']];event=payload['events'][item['event_index']]
        if event['key']!=item['key']:raise ValueError('event index drift')
        return item,payload,event

    def get(self,index,device='cpu'):
        item,payload,event=self.event(index);row=self.rows[tuple(item['key'])];array=self.arrays(tuple(item['key']))
        query=payload['observations'][tuple(event['query_source'])];call=payload['calls'][event['call_index']]
        native_scores=torch.tensor(event['native_track_scores'],device=device,dtype=torch.float32)
        histories=[];pairs=[]
        for position,track in enumerate(event['candidate_tracks']):
            snapshot=call['histories'][track];history=[payload['observations'][tuple(source)] for source in snapshot['sources']]
            ordered=sorted(history,key=lambda obs:(obs['frame'],obs['camera'],obs['source']))
            values=torch.stack([obs['feature'].float().clone() for obs in ordered]).to(device)
            times=torch.tensor([obs['frame'] for obs in ordered],device=device,dtype=torch.float32)
            cameras=torch.tensor([obs['camera'] for obs in ordered],device=device)
            if bool((times>query['frame']).any()):raise ValueError('future history in model inputs')
            summary=torch.nn.functional.normalize(values,dim=-1).mean(0)
            current=query['feature'].float().to(device)
            cosine=torch.nn.functional.cosine_similarity(current[None],summary[None])[0]
            gap=query['frame']-int(times[-1]);same=float(int(cameras[-1])==query['camera'])
            pairs.append(torch.stack((native_scores[position].log1p(),native_scores[position]/native_scores.max().clamp_min(1e-8),
                cosine,native_scores.new_tensor(float(gap)).log1p(),native_scores.new_tensor(same),
                native_scores.new_tensor(float(snapshot['observed_length'])).log1p(),native_scores.new_tensor(event['relative_margin']),
                (cameras!=query['camera']).float().mean())))
            histories.append(dict(features=values,times=times,cameras=cameras,observed_length=snapshot['observed_length']))
        features=dict(current=query['feature'].float().clone().to(device),histories=histories,pair=torch.stack(pairs),
            time=query['frame'],camera=query['camera'],native_scores=native_scores,
            layers=[torch.from_numpy(array['layer_'+str(layer)]).float().to(device) for layer in (4,11,17,23)],
            attention=torch.from_numpy(array['attention']).float().to(device),xy=torch.from_numpy(array['xy']).float().to(device),
            inside=torch.from_numpy(array['inside']).bool().to(device),roi=torch.from_numpy(array['roi']).float().to(device))
        positives=[track in event['positive_tracks'] for track in event['candidate_tracks']]
        null_positive=event['known_unmatched_within_all_recorded_history'] and not any(positives)
        target=dict(positive=torch.tensor(positives+[null_positive],device=device),
            known=torch.tensor([track not in event['ambiguous_tracks'] for track in event['candidate_tracks']]+[True],device=device),
            loss_eligible=event['loss_eligible'],query_gt_known=event['query_gt_identity'] is not None,
            any_history_positive=event['all_history_positive'],top8_positive=event['top8_positive'],
            ambiguous_indices=[i for i,track in enumerate(event['candidate_tracks']) if track in event['ambiguous_tracks']],
            key=item['key'],scope=item['scope'],scene=item['scene'],actual_primary_state=event['actual_primary_state'],
            native_top1_state=item['native_top1_state'],known_unmatched=null_positive,patch_bin=row['patch_bin'])
        return features,target

    def native_prior_scores(self,index):
        _,payload,event=self.event(index);query=payload['observations'][tuple(event['query_source'])];call=payload['calls'][event['call_index']]
        logits=call['logits'][event['query_row']].float().clone()
        camera=torch.tensor(call['reference_cameras']);time=torch.tensor(call['reference_times']);finite=torch.tensor(call['reference_finite'])
        legal=(camera!=query['camera'])&((time-query['frame']).abs()<=20)&finite
        logits+=legal.float()*2.
        pieces=[];first=0
        for count in call['reference_block_sizes']:
            block=logits[first:first+count];pieces.append(torch.cat((block,block.new_zeros(1))).softmax(0)[:-1]);first+=count
        if first!=len(logits):raise ValueError('native probability-block mismatch')
        probability=torch.cat(pieces);ids=torch.tensor([ref['track'] for ref in call['references']])
        scores=torch.stack([probability[ids==track].sum() for track in event['candidate_tracks']])
        return torch.cat((scores,scores.new_zeros(1)))


def association_losses(logits,target):
    positive=target['positive'];known=target['known']
    if positive.shape!=logits.shape or known.shape!=logits.shape or bool((positive&~known).any()):raise ValueError('invalid positive/unknown/NULL label contract')
    if not target['loss_eligible'] or not positive.any():return None
    assoc=torch.logsumexp(logits[known],0)-torch.logsumexp(logits[positive],0)
    negative=known&~positive
    rank=torch.relu(logits[negative].max()-logits[positive].max()+.2) if negative.any() else assoc*0
    return assoc+.25*rank,assoc,rank

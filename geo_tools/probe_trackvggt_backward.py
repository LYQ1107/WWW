"""One real TRAIN candidate relation/backward pass, zero optimizer updates."""
import numpy as np
import torch
from common import *

NEW=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')


def main():
    from gtr.geometry.track_query_encoder import TrackQueryEncoder
    from gtr.geometry.identity_guided_sampler import IdentityGuidedSampler
    from gtr.geometry.track_qd_teacher import TrackQDTeacher
    results=[]
    for backbone in ('vggt','dinov2'):
        folder=NEW/'tokens'/f'probe_{backbone}_scene4_v1';manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['limited_probe'] and manifest['scene']==4
        row=manifest['rows'][0];assert sha256(row['file'])==row['sha256']
        arrays=np.load(row['file']);payload=torch.load(row['event_file'],map_location='cpu');event=payload['events'][row['event_index']]
        if not event['positive_tracks']:raise ValueError('real pilot query lacks a known positive; do not fabricate one')
        query=payload['observations'][tuple(event['query_source'])];call=payload['calls'][event['call_index']]
        channels=row['channels'];bottleneck=228 if backbone=='dinov2' else None
        seed_all(20261015);encoder=TrackQueryEncoder(payload['feature_dim']);sampler=IdentityGuidedSampler(channels,projection_hidden=bottleneck)
        qd=TrackQDTeacher([channels]*4,projection_hidden=bottleneck)
        layers=[torch.from_numpy(arrays['layer_'+str(layer)].copy()).float() for layer in (4,11,17,23)]
        inside=torch.from_numpy(arrays['inside'].copy());attention=torch.from_numpy(arrays['attention'].copy()).float()
        logits=[]
        for index,track in enumerate(event['candidate_tracks']):
            snapshot=call['histories'][track];history=[payload['observations'][tuple(source)] for source in snapshot['sources']]
            native=torch.stack([obs['feature'].float() for obs in history]);times=torch.tensor([obs['frame'] for obs in history]).float();cameras=torch.tensor([obs['camera'] for obs in history])
            q=encoder(native,times,cameras,query['frame'],query['camera'],snapshot['observed_length'])
            score,_=sampler.scores(layers[2],attention,q,inside,'C')
            weights=score.softmax(0);pooled=[(features*weights[:,None]).sum(0,keepdim=True) for features in layers]
            latest=max(obs['frame'] for obs in history);native_current=query['feature'].float()
            cosine=torch.nn.functional.cosine_similarity(native_current[None],native.mean(0)[None])[0]
            pair=torch.tensor([np.log1p(event['native_track_scores'][index]),1.,float(cosine),np.log1p(query['frame']-latest),
                float(history[-1]['camera']==query['camera']),np.log1p(snapshot['observed_length']),event['relative_margin'],
                sum(obs['camera']!=query['camera'] for obs in history)/len(history)])
            value,_=qd(q,pooled,pair);logits.append(value)
        logits=torch.cat((torch.stack(logits),torch.zeros(1)))
        positive=torch.tensor([track in event['positive_tracks'] for track in event['candidate_tracks']]+[False])
        known=torch.tensor([track not in event['ambiguous_tracks'] for track in event['candidate_tracks']]+[True])
        loss=torch.logsumexp(logits[known],0)-torch.logsumexp(logits[positive],0)
        loss.backward();components=dict(track_query=encoder.mlp[0].weight,ag_query=sampler.query.weight,
            ag_key=sampler.key[0].weight if backbone=='dinov2' else sampler.key.weight,
            see_query=qd.see.condition[0].weight,layer_weight_head=qd.see.task_head[0].weight,
            cross_attention=qd.decoder.layers[0].multihead_attn.in_proj_weight)
        components.update({f'projection_{i}':proj[0].weight for i,proj in enumerate(qd.projections)})
        norms={name:float(param.grad.norm()) if param.grad is not None else None for name,param in components.items()}
        assert torch.isfinite(loss) and all(value is not None and np.isfinite(value) and value>0 for value in norms.values())
        results.append(dict(backbone=backbone,event_key=event['key'],native_feature_dim=payload['feature_dim'],token_shape=list(layers[0].shape),
            listwise_loss=float(loss),gradient_norms=norms,trainable_parameters=sum(p.numel() for m in (encoder,sampler,qd) for p in m.parameters()),
            token_file=row['file'],token_sha256=row['sha256'],event_file=row['event_file'],event_sha256=sha256(row['event_file'])))
    report=dict(status='PASS_REAL_TRAIN_RELATION_BACKWARD',optimizer_updates=0,rows=results,
        scope='one real scene4 query with legal observed histories and post-run TRAIN supervision, soft AG relevance and all QD projections receive gradients; no fitted checkpoint, validation score or teacher success',provenance=provenance())
    save_json(ROOT/'results/trackvggt_real_backward_v1.json',report)
    print(json.dumps(report['rows']))


if __name__=='__main__':main()

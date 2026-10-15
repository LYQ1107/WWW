"""History-pair ranking diagnostic on TRAIN, with no unmatched calibration loss."""
import argparse,time
import numpy as np
import torch
from common import *
from gtr.geometry.lightweight_head import LowRankIdentityHead


def history_clauses(observations,labels,events,max_age=20):
    clauses=[]
    for event in events:
        for position,query in enumerate(event['query']):
            if query<0 or labels[query]<0:continue
            current=observations[query];groups={}
            for column,reference in enumerate(event['reference']):
                if reference<0 or labels[reference]<0:continue
                past=observations[reference];age=current['frame']-past['frame']
                if current['camera']==past['camera'] or not 0<age<=max_age:continue
                groups.setdefault((past['frame'],past['camera']),[]).append((column,reference))
            for entries in groups.values():
                columns,references=zip(*entries);positive=np.asarray([labels[r]==labels[query] for r in references])
                if not positive.any() or positive.all():continue
                clauses.append(dict(query=query,reference=list(references),base=event['logits'][position,list(columns)].float(),
                                    positive=torch.as_tensor(positive),query_frame=current['frame']))
    return clauses


def ranking_loss(candidate,positive):
    # No unmatched column. A row-common confidence bonus cannot improve this
    # objective; only relative candidate evidence can change the loss/ranking.
    return torch.logsumexp(candidate,0)-torch.logsumexp(candidate[positive],0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    out=RUNTIME/'heads'/args.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(1)
    datasets={};sources={};paired_provenance=None
    for video in [5,8,12]:
        run=RUNTIME/'history_train'/f'history_train_keyframe_v1_v{video}';manifest=run/'manifest.json';info=json.loads(manifest.read_text())
        if info['source_split']!='association_train' or info['video_id']!=video or not info['neutral_prefix40_exact'] or info['encoder_failures']:
            raise ValueError('invalid TRAIN collection gate')
        paired={key:info['provenance'][key] for key in ['seed','checkpoint_sha256','config_sha256','annotation_sha256']}
        if paired_provenance is not None and paired!=paired_provenance:raise ValueError('unpaired TRAIN collections')
        paired_provenance=paired
        for name,value in info['artifacts'].items():
            if sha256(run/name)!=value:raise ValueError('changed TRAIN artifact')
        metadata=json.loads((run/'observations.json').read_text());arrays=np.load(run/'targets.npz')
        if any(not np.isfinite(arrays[kind]).all() or len(arrays[kind])!=len(metadata) for kind in ['gmt','dinov2','vggt']):
            raise ValueError('nonfinite or misaligned target archive')
        clauses=history_clauses(metadata,arrays['labels'],torch.load(run/'events.pt',map_location='cpu'))
        if not clauses:raise ValueError('no eligible historical positive/negative clauses')
        for clause in clauses:
            for key in ('base','positive'):clause[key]=clause[key].to(device)
            clause['reference']=torch.tensor(clause['reference'],device=device)
        datasets[video]=dict(vectors={kind:torch.tensor(arrays[kind].astype(np.float32),device=device) for kind in ['gmt','dinov2','vggt']},clauses=clauses)
        sources[str(manifest)]=sha256(manifest)
    fit=[(video,clause) for video in [5,8] for clause in datasets[video]['clauses']]
    heldout=[(12,clause) for clause in datasets[12]['clauses']]
    def candidate(model,kind,video,clause):
        vectors=datasets[video]['vectors'];query=clause['query'];refs=clause['reference']
        foundation=[None,None] if kind=='gmt' else [vectors[kind][query:query+1],vectors[kind][refs]]
        evidence=model(vectors['gmt'][query:query+1],vectors['gmt'][refs],*foundation)[0]
        return clause['base']+2*evidence
    @torch.no_grad()
    def evaluate(model=None,kind=None,items=heldout):
        losses=[];correct=[]
        for video,clause in items:
            score=clause['base'] if model is None else candidate(model,kind,video,clause)
            losses.append(float(ranking_loss(score,clause['positive'])))
            correct.append(bool(clause['positive'][score.argmax()]))
        return dict(clauses=len(correct),loss=float(np.mean(losses)),recall1=float(np.mean(correct)),correct=correct)
    native=evaluate();results=[];curves=[];started=time.perf_counter()
    for kind in ['gmt','dinov2','vggt']:
        for seed in [20261014,20261015,20261016]:
            warm=RUNTIME/'heads/low_rank_three_backbones_train_v1'/f'{kind}_seed{seed}.pt';checkpoint=torch.load(warm,map_location='cpu')
            if checkpoint['source_split']!='association_train' or checkpoint['backbone']!=kind or checkpoint['seed']!=seed:raise ValueError('invalid warm start')
            seed_all(seed);rng=np.random.default_rng(seed);model=LowRankIdentityHead(**checkpoint['settings']).to(device)
            model.load_state_dict(checkpoint['state_dict'],strict=True);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            before=evaluate(model,kind);best=float('inf');destination=out/warm.name
            for step in range(1,2001):
                video,clause=fit[int(rng.integers(len(fit)))];optimizer.zero_grad(set_to_none=True)
                loss=ranking_loss(candidate(model,kind,video,clause),clause['positive'])
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite historical ranking loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
                with torch.no_grad():model.scale.clamp_(.01,10.)
                if step%500==0:
                    metrics=evaluate(model,kind);curves.append(dict(backbone=kind,seed=seed,step=step,**{k:v for k,v in metrics.items() if k!='correct'}))
                    if metrics['loss']<best:
                        best=metrics['loss'];selected=dict(checkpoint,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            step=step,history_objective='past other-camera per-reference-frame ranking; no unmatched logit',
                            history_training_sources=sources,warm_start_checkpoint_sha256=sha256(warm),
                            warm_start_sample_manifest_sha256=checkpoint.get('sample_manifest_sha256'),
                            history_training_protocol_sha256=sha256(ROOT/'manifests/history_training_queue_v1.json'))
                        selected.pop('sample_manifest_sha256',None)
                        torch.save(selected,destination)
            model.load_state_dict(torch.load(destination,map_location=device)['state_dict'],strict=True);after=evaluate(model,kind)
            after['fixes_native']=sum(not a and b for a,b in zip(native['correct'],after['correct']))
            after['harms_native']=sum(a and not b for a,b in zip(native['correct'],after['correct']))
            results.append(dict(backbone=kind,seed=seed,before=before,after=after,checkpoint=str(destination),checkpoint_sha256=sha256(destination),
                parameters=sum(p.numel() for p in model.parameters())))
            print(json.dumps(dict(backbone=kind,seed=seed,recall1=after['recall1'],fixes=after['fixes_native'],harms=after['harms_native'])),flush=True)
    primary={row['backbone']:row['after'] for row in results if row['seed']==20261014}
    gate=primary['vggt']['recall1']>max(native['recall1'],primary['gmt']['recall1'],primary['dinov2']['recall1']) and primary['vggt']['fixes_native']>primary['vggt']['harms_native']
    report=dict(status='HISTORY_TRAIN_DIAGNOSTIC_COMPLETE',unique_vggt_primary_gate=gate,
        gate_rule='primary-seed VGGT heldout TRAIN12 history Recall1 strictly exceeds native GMT and newly trained matched GMT/DINO controls, with more fixes than harms; no DEV seed/weight selection',
        next_action='predefined primary head causal DEV verification is permitted, not yet run' if gate else 'no unique positive heldout TRAIN signal; do not expand this branch into DEV tuning',
        fit_video_ids=[5,8],validation_video_ids=[12],steps=2000,seeds=[20261014,20261015,20261016],
        source_split='association_train',used_cache=True,speed_eligible=False,seconds=time.perf_counter()-started,
        scope='bounded TRAIN-only frozen target-vector/history-logit cache, same-time/same-camera/future/stale/unknown pairs excluded; each clause has known positive and negative candidates from one past opposite-camera frame',
        loss='logsumexp over native proposal logits plus bounded weight2 learned evidence minus positive logsumexp; unmatched column omitted so uniform candidate offset cancels',
        selection_rule='minimum heldout TRAIN12 loss every500 updates; all three seeds retained; unchanged matched architectures and bounded preactivation score adapter',
        limitations='conditional query/reference-frame pair ranking on fixed sparse TRAIN history; one query can contribute several past-frame clauses; TRAIN12 is reused for checkpoint selection, not an unbiased test; not whole-GTA/Hungarian training, full tracking, retrieval, causal runtime or independent backbone generalization',
        native=native,clauses_by_video={str(video):len(value['clauses']) for video,value in datasets.items()},results=results,curves=curves,source_manifests=sources,provenance=provenance())
    save_json(out/'manifest.json',report);save_json(RUNTIME/'matrix_reports'/f'{args.tag}.json',report)


if __name__=='__main__':main()

"""Materialize real A(previous)->B(current)->A(current) teacher relations.

Only the locked observation-selected128 cycles are used. Missing native AB
pairs are omitted from model inputs; GT masks are separate loss arguments.
A(current) remains one selected detection plus NULL, explicitly bounded.
"""
import gzip
import collections
import numpy as np
import torch
from common import *
from gtr.geometry.temporal_identity_cycle import temporal_cycle_loss


def relation_inputs(data,scene,query,nodes,scores,normalizer,margin,observations,device='cpu'):
    if len(nodes)!=len(scores) or any(value is None for value in scores):raise ValueError('missing native relation must be omitted, not fabricated')
    key=(scene,*query['source']);arrays=data.arrays(key);native=torch.tensor(scores,device=device,dtype=torch.float32)
    histories=[];pairs=[];current=query['feature'].float().clone().to(device)
    for node,score in zip(nodes,native):
        history=[observations[tuple(source)] for source in node['history_sources']]
        history=sorted(history,key=lambda o:(o['frame'],o['camera'],o['source']))
        if not history:raise ValueError('missing actual cycle history')
        values=torch.stack([o['feature'].float().clone() for o in history]).to(device)
        times=torch.tensor([o['frame'] for o in history],device=device,dtype=torch.float32)
        cameras=torch.tensor([o['camera'] for o in history],device=device)
        if bool((times>query['frame']).any()):raise ValueError('future cycle history')
        summary=torch.nn.functional.normalize(values,dim=-1).mean(0)
        cosine=torch.nn.functional.cosine_similarity(current[None],summary[None])[0]
        gap=query['frame']-int(times[-1]);same=float(int(cameras[-1])==query['camera'])
        pairs.append(torch.stack((score.log1p(),score/score.new_tensor(normalizer).clamp_min(1e-8),cosine,
            score.new_tensor(float(gap)).log1p(),score.new_tensor(same),score.new_tensor(float(node['observed_length'])).log1p(),
            score.new_tensor(margin),(cameras!=query['camera']).float().mean())))
        histories.append(dict(features=values,times=times,cameras=cameras,observed_length=node['observed_length']))
    return dict(current=current,histories=histories,pair=torch.stack(pairs) if pairs else torch.empty((0,8),device=device),
        time=query['frame'],camera=query['camera'],native_scores=native,
        layers=[torch.from_numpy(arrays['layer_'+str(k)].copy()).float().to(device) for k in (4,11,17,23)],
        attention=torch.from_numpy(arrays['attention'].copy()).float().to(device),xy=torch.from_numpy(arrays['xy'].copy()).float().to(device),
        inside=torch.from_numpy(arrays['inside'].copy()).bool().to(device),roi=torch.from_numpy(arrays['roi'].copy()).float().to(device))


class NativeCycleData:
    def __init__(self,data):
        self.data=data
        if data.cycle_overlay:plan=json.loads(data.cycle_overlay.read_text())
        else:plan=data.cohort
        flag='in_V3_cycle_subset' if data.cohort.get('cohort_version',2)==3 else 'in_V2_cycle_subset'
        self.specs=[row for row in plan['cycle_rows'] if row[flag]]
        if len(self.specs)!=128:raise ValueError('locked bounded cycle membership differs')
        fit={tuple(data.items[i]['key']) for i in data.fit}
        if any(tuple(row['key']) not in fit for row in self.specs):raise ValueError('cycle query outside original fit split')
        wanted=collections.defaultdict(set)
        for row in self.specs:
            event=data.payloads[row['event_file']]['events'][row['event_index']]
            graph=event['cycle']
            for index in row['B_indices']:wanted[row['key'][0]].add(tuple(graph['B_current'][index]['source']))
        self.native_margins={}
        for scene,keys in wanted.items():
            root=Path('/data1/liuyeqiang/WWW_trackvggt_runtime/full_train_audit_v1')/f'scene{scene}_full_v1'
            manifest=json.loads((root/'manifest.json').read_text());path=root/'rankings.jsonl.gz'
            if sha256(path)!=manifest['artifacts'][path.name]:raise ValueError('original native ranking provenance changed')
            data.source_artifacts[str(path)]=sha256(path)
            with gzip.open(path,'rt') as stream:
                for line in stream:
                    item=json.loads(line)
                    for source,margin in zip(item['queries'],item['relative_margin']):
                        if tuple(source) in keys:self.native_margins[(scene,*source)]=margin
            if any((scene,*source) not in self.native_margins for source in keys):raise ValueError('actual earlier B margin absent')

    def get(self,index,device='cpu'):
        spec=self.specs[index];scene=spec['key'][0];payload=self.data.payloads[spec['event_file']]
        event=payload['events'][spec['event_index']];graph=event['cycle'];target=payload['cycle_targets'][tuple(event['key'])]
        observations=payload['observations'];query=observations[tuple(event['query_source'])]
        a=graph['A_previous'];b_indices=spec['B_indices'];b=[graph['B_current'][i] for i in b_indices]
        score_map=dict(zip(event['candidate_tracks'],event['native_track_scores']));ab=[];observed=torch.zeros((len(a),len(b)),dtype=torch.bool,device=device)
        for bi,(original,node) in enumerate(zip(b_indices,b)):
            scores=graph['AB_native_scores_by_B_then_A'][original];indices=[i for i,score in enumerate(scores) if score is not None]
            if not indices:continue
            b_query=observations[tuple(node['source'])]
            if b_query['frame']!=graph['current_time'] or b_query['camera']==query['camera']:raise ValueError('illegal current B source')
            inputs=relation_inputs(self.data,scene,b_query,[a[i] for i in indices],[scores[i] for i in indices],
                graph['B_native_score_normalizers'][original],self.native_margins[(scene,*node['source'])],observations,device)
            ab.append(dict(B_index=bi,A_indices=indices,features=inputs));observed[indices,bi]=True
        for node in a:
            if any(observations[tuple(s)]['frame']>graph['previous_time'] for s in node['history_sources']):raise ValueError('direct A history is not strictly past')
        aa=relation_inputs(self.data,scene,query,a,[score_map[n['track']] for n in a],graph['A_current_native_score_normalizer'],event['relative_margin'],observations,device)
        ba=relation_inputs(self.data,scene,query,b,[score_map[n['track']] for n in b],graph['A_current_native_score_normalizer'],event['relative_margin'],observations,device)
        legal=torch.tensor([[row[i] for i in b_indices] for row in target['legal_AB_pairs']],dtype=torch.bool,device=device)&observed
        known_b=torch.tensor([target['known_B_columns'][i] for i in b_indices],dtype=torch.bool,device=device)
        visible=torch.tensor([identity is not None and identity==target['current_A_GT'] and any(bool(legal[i,j]) and target['B_current_GT'][original]==identity for j,original in enumerate(b_indices))
            for i,identity in enumerate(target['A_past_GT'])],dtype=torch.bool,device=device)
        if visible.cpu().tolist()!=spec['visible_rows_after_fixed_B_subset']:raise ValueError('locked post-parity visibility mask differs')
        features=dict(ab=ab,aa=aa,ba=ba,A_count=len(a),B_count=len(b),previous_time=graph['previous_time'],current_time=graph['current_time'])
        masks=dict(legal_AB=legal,known_B=known_b,visible_A=visible)
        return features,masks


def forward_cycle(model,features,masks):
    device=features['aa']['current'].device;a,b=features['A_count'],features['B_count']
    scores=torch.full((a,b),-torch.inf,device=device)
    for item in features['ab']:
        values,_=model(item['features'],soft=False)
        scores[item['A_indices'],item['B_index']]=values[:-1]
    # GT masks normalize only the *training loss* relation space. The model
    # inputs and all observed pair forwards were constructed without these masks.
    ab=torch.cat((scores.masked_fill(~masks['legal_AB'],-torch.inf),torch.zeros((a,1),device=device)),1).softmax(1)
    aa_values,_=model(features['aa'],soft=False);ba_values,_=model(features['ba'],soft=False)
    aa=torch.stack((aa_values[:-1],torch.zeros(a,device=device)),1).softmax(1)
    ba_logits=torch.stack((ba_values[:-1],torch.zeros(b,device=device)),1)
    ba_logits[:,0]=ba_logits[:,0].masked_fill(~masks['known_B'],-torch.inf);ba=ba_logits.softmax(1)
    loss=temporal_cycle_loss(ab,ba,aa,masks['visible_A'],features['previous_time'],features['current_time'])
    return loss,dict(P_AB=ab,P_BA=ba,P_AA=aa,visible_rows=int(masks['visible_A'].sum()))

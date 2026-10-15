"""Same-checkpoint variance gate ablation; no retraining/model selection.

Epoch1 is explicitly posthoc diagnosis. The same protocol is locked before
future fixed epochs3/5; feature-error calibration is not evidence of identity
benefit or proof that the gate ever acts on an association score.
"""
import argparse,csv,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from cv_pil_dev12_shards import NativeShardAssets
from cv_pil_v2_model import tensorize
from cv_pil_native_handoff import safe_torch_load
from pirga_candidate_state import candidate_state
from pirga_innovation import InnovationHead
from pirga_indexed_store import sha
from run_pirga_phase1_replay import primary_solve


def execute(pins_path,out):
    if out.exists():raise FileExistsError(out)
    pins=json.loads(pins_path.read_text())
    if sha(__file__)!=pins['source_sha256']:raise ValueError('Gate diagnostic source changed')
    p2=pins['phase2_result']
    if sha(p2['pins_path'])!=p2['pins_sha256'] or sha(p2['query_csv'])!=p2['query_csv_sha256']:raise ValueError('Frozen endpoint changed')
    source=json.loads(Path(p2['pins_path']).read_text());epoch=source['epoch'];spec=source['heads']['A4_uncertainty'];protocol=source['protocol']
    if sha(spec['path'])!=spec['sha256'] or sha(protocol['path'])!=protocol['sha256']:raise ValueError('Frozen head/protocol changed')
    p=json.loads(Path(protocol['path']).read_text());repo=Path(__file__).resolve().parents[1]
    for name,digest in p['source_code_sha256'].items():
        if sha(repo/name)!=digest:raise ValueError('Frozen training dependency changed')
    state=safe_torch_load(spec['path'])
    if state['arm']!='A4_uncertainty' or state['epoch']!=epoch or state['step']!=113415*epoch or state['protocol_sha256']!=protocol['sha256']:raise ValueError('Invalid fixed head')
    torch.set_num_threads(1);start=time.monotonic();head=InnovationHead('A4_uncertainty');head.load_state_dict(state['state_dict'],strict=True);head.eval();assets=NativeShardAssets(**source['release'])
    if assets.engineering_only:raise ValueError('No prefix gate audit')
    original={};off={};counts=dict(candidate_call_states=0,nonempty_history_candidate_call_states=0,variance_above1=0,nonempty_history_variance_above1=0);max_variance=0.;max_delta=0.;changed_score_edges=0
    with torch.no_grad():
        for cid,c in assets.calls.items():
            if c['stage']=='run_memory_tracker' or not c['queries']:continue
            contexts={(assets.observations[tuple(q)]['frame'],assets.observations[tuple(q)]['camera'],assets.observations[tuple(q)]['image_id']) for q in c['queries']}
            if len(contexts)!=1:raise ValueError('Mixed native context')
            actor=assets.actor((12,*c['queries'][0]));b=tensorize(actor);lags=torch.tensor([r['lag'] for rows in actor['past_histories'] for r in rows],dtype=torch.float32)
            st=candidate_state(b.histories,b.owners,lags,b.cross_view,b.destination_camera,len(b.native_scores),p['horizon_boundaries']);mu,lv=head.predict(st);variance=lv.exp();gate=torch.exp(-lv.clamp_min(0));on_scores=[];off_scores=[]
            counts['candidate_call_states']+=len(variance);counts['nonempty_history_candidate_call_states']+=int((st.counts>0).sum());counts['variance_above1']+=int((variance>1).sum());counts['nonempty_history_variance_above1']+=int(((variance>1)&(st.counts>0)).sum())
            if len(variance):max_variance=max(max_variance,float(variance.max()))
            for row,key in enumerate(c['query_feature_keys']):
                q=F.normalize(assets.feature(key).float().clone(),dim=0);base=c['track_scores'][row]+2*(st.mean@q)
                innovation=(mu@q-F.normalize(st.mean,dim=-1)@q).clamp(-.25,.25);delta=2*(st.counts/(st.counts+1))*innovation
                on=base+gate*delta;ungated=base+delta
                # Original source multiplies support,gate and innovation in
                # this exact FP32 order; use it for control attestation.
                on=base+2*(st.counts/(st.counts+1))*gate*innovation
                on_scores.append(on);off_scores.append(ungated)
                if len(on):max_delta=max(max_delta,float(abs(on-ungated).max()));changed_score_edges+=int((on!=ungated).sum())
            a=primary_solve(torch.stack(on_scores),c['track_ids'],c['thresholds']);v=primary_solve(torch.stack(off_scores),c['track_ids'],c['thresholds'])
            for row,key in enumerate(c['queries']):original[(12,*key)]=a[row];off[(12,*key)]=v[row]
    # All actions generated before parsing the old reporting CSV. The audit
    # uses only its physical keys and frozen ON accepted IDs, never GT labels.
    with Path(p2['query_csv']).open() as f:reference={tuple(json.loads(r['key'])):r for r in csv.DictReader(f)}
    if set(reference)!=set(original):raise ValueError('Scope drift')
    for key,value in original.items():
        if reference[key]['A4_uncertainty_accepted_id']!=('' if value is None else str(value)):raise ValueError('ON action control drift')
    changed=[list(k) for k in original if original[k]!=off[k]]
    result=dict(status='SAME_CHECKPOINT_UNCERTAINTY_GATE_DIAGNOSTIC_ONLY',epoch=epoch,preregistration=pins['diagnostic_registration'],counts=counts,max_predicted_variance=max_variance,
        max_absolute_ON_OFF_score_difference=max_delta,changed_score_edges=changed_score_edges,primary_action_changes=len(changed),changed_keys=changed,
        original_ON_actions_exact_full_scope=True,queries=len(original),weights_unchanged=True,training_updates=0,GPU_wall_hours=0,wall_seconds=time.monotonic()-start,
        prediction_variance_is_identity_probability=False,scientific_success=False,
        interpretation='Zero score/action change means the inference gate explains no identity benefit at this endpoint. Training with heteroscedastic loss may still change the mean predictor. Never conflate those mechanisms; no hyperparameter adjustment follows this audit.',
        interval_warning='Distinct native candidate-call states share identities and observations; edge counts are not independent successes.',pins_sha256=sha(pins_path))
    out.mkdir(parents=True);(out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');(out/'INPUT_PINS.json').write_text(json.dumps(pins,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pins',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();execute(a.pins,a.out)

"""Actual frozen128-cycle materialization and one fitted-AG gradient probe."""
import torch
from common import *
from trackvggt_data import TrackEventData
from trackvggt_cycle_features import NativeCycleData,forward_cycle
from evaluate_teacher_pairs import build_model


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');cohort=runtime/'cohorts/hard_events_expanded_v3/cohort.json'
    data=TrackEventData('vggt',cohort_path=cohort);cycles=NativeCycleData(data);counts=[]
    for i in range(len(cycles.specs)):
        features,masks=cycles.get(i)
        counts.append(dict(key=cycles.specs[i]['key'],A=features['A_count'],B=features['B_count'],observed_AB=sum(len(r['A_indices']) for r in features['ab']),
            legal_AB=int(masks['legal_AB'].sum()),visible_A=int(masks['visible_A'].sum()),
            A_current_scope='one actual selected detection plus NULL'))
    # GT selects a visible chain only for a finite implementation/gradient test,
    # never the128-cycle formal scope or teacher-fit membership.
    index=next(i for i,r in enumerate(counts) if r['visible_A'])
    checkpoint=runtime/'models/ag_vggt_C_mean_k24_seed20261015_v3_1000_v1/checkpoint.pt';state=torch.load(checkpoint,map_location='cpu')
    if state['cohort_sha256']!=sha256(cohort):raise ValueError('fitted probe source differs')
    model=build_model(state['settings']).to('cuda');model.load_state_dict(state['model'],strict=True);model.train()
    features,masks=cycles.get(index,'cuda');loss,matrices=forward_cycle(model,features,masks)
    loss.backward();norms={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
    if not torch.isfinite(loss) or any(not np.isfinite(n) for n in norms.values()) or not any(n>0 for n in norms.values()):raise ValueError('actual cycle has no finite trainable path')
    proof=dict(status='PASS_REAL_V3_CYCLE_FEATURES_AND_FITTED_AG_GRADIENT',cycles=len(counts),visible_cycles=sum(r['visible_A']>0 for r in counts),
        bounded_B_max=max(r['B'] for r in counts),all_sources_actual_and_causal=True,cycle_membership_hash=hashlib.sha256(json.dumps(cycles.specs,sort_keys=True).encode()).hexdigest(),
        rows=counts,probe_key=counts[index]['key'],probe_loss=float(loss.detach()),probe_gradient_norms=norms,
        matrix_shapes={k:list(v.shape) for k,v in matrices.items() if isinstance(v,torch.Tensor)},checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        model_fitting_steps=0,scope='Implementation/data/gradient gate on fitted AG, not trained QD or temporal benefit. Formal no-cycle/cycle continuation still required.',
        source_artifacts=data.source_artifacts,provenance=provenance())
    save_json(ROOT/'results/trackvggt_real_cycle_features_v1.json',proof);print(json.dumps({k:v for k,v in proof.items() if k not in ('rows','source_artifacts','provenance')}))


if __name__=='__main__':
    import numpy as np
    main()

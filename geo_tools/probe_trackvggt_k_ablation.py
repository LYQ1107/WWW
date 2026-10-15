"""Prespecified inference K16/24/32, identical fitted AG-C weights/validation."""
import torch
from common import *
from trackvggt_data import TrackEventData
from evaluate_teacher_pairs import build_model,evaluate


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');checkpoint=runtime/'models/ag_vggt_C_mean_k24_seed20261015_v3_1000_v1/checkpoint.pt'
    state=torch.load(checkpoint,map_location='cpu');data=TrackEventData('vggt',state['cohort_path']);results=[]
    if state['cohort_sha256']!=sha256(data.cohort_path):raise ValueError('fitted K ablation cohort differs')
    for k in [16,24,32]:
        settings=dict(state['settings'],k=k);model=build_model(settings);model.load_state_dict(state['model'],strict=True)
        result=evaluate(model,data,data.validation,'cpu');results.append(dict(k=k,metrics=result['metrics'],rows=result['rows'],seconds=result['seconds']))
    prior=json.loads((checkpoint.parent/'validation.json').read_text())
    fields=['key','prediction_index','state','positive_first_rank_including_null','matched_candidate_first_rank']
    if [[row[field] for field in fields] for row in results[1]['rows']]!=[[row[field] for field in fields] for row in prior['rows']]:raise ValueError('K24 CPU/GPU discrete ranking control changed; retain evidence before interpreting K')
    max_error=max(abs(x-y) for row,old in zip(results[1]['rows'],prior['rows']) for x,y in zip(row['logits'],old['logits']))
    proof=dict(status='COMPLETE_FITTED_K_INFERENCE_ABLATION',source_checkpoint_sha256=sha256(checkpoint),training_K=24,
        evaluation_K=[16,24,32],same_fitted_weights=True,model_fitting_steps=0,rows=results,
        K24_CPU_GPU_discrete_ranking_exact=True,K24_CPU_GPU_max_abs_logit_error=max_error,
        scope='K at inference only, fixed relevance/model source and original300. Not independently retrained K models or a selected final tracking method.',provenance=provenance())
    save_json(ROOT/'results/trackvggt_ag_K_ablation_v1.json',proof);print(json.dumps(dict(status=proof['status'],metrics=[dict(k=r['k'],Recall1=r['metrics']['Recall1']) for r in results])))


if __name__=='__main__':main()

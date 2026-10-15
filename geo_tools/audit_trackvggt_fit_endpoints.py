"""Retrospective hard-inference ranks on saved150/1000 fitted endpoints.

No optimization, model selection or reconstructed intermediate curve. All
registered teacher cases are considered; incomplete cases remain pending.
"""
import torch
from common import *
from trackvggt_data import TrackEventData
from trackvggt_crop_data import CropEventData
from prepare_trackvggt_crop_comparison import select_fit
from evaluate_teacher_pairs import build_model,evaluate


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');out=runtime/'diagnostics/fit_endpoints_v1'
    out.mkdir(parents=True,exist_ok=True);datasets={};reports=[];pending=[]
    plans=['trackvggt_phase2_ag_fit_v1','trackvggt_v3_ag_fit_v1','trackvggt_qd_fit_v1','trackvggt_crop_fit_v1','trackvggt_qd_native_control_v1']
    for plan in plans:
        value=json.loads((ROOT/'manifests'/f'{plan}.json').read_text())
        for stages in value['queues'].values():
            for stage in stages:
                tag=stage['id'];source=runtime/'models'/tag;manifest=source/'manifest.json';target=out/(tag+'.json')
                if not manifest.exists():pending.append(tag);continue
                m=json.loads(manifest.read_text())
                if m.get('status')!='COMPLETE_FINITE_TEACHER_FIT_AND_FIXED_VALIDATION':pending.append(tag);continue
                if target.exists():
                    prior=json.loads(target.read_text())
                    if prior['source_checkpoint_sha256']!=m['checkpoint_sha256']:raise ValueError('diagnostic source changed; do not replace')
                    reports.append(dict(tag=tag,path=str(target),sha256=sha256(target),metrics=prior['fit_metrics']));continue
                checkpoint=source/'checkpoint.pt'
                if sha256(checkpoint)!=m['checkpoint_sha256']:raise ValueError('fitted endpoint checkpoint changed')
                state=torch.load(checkpoint,map_location='cpu');settings=state['settings'];key=(settings['backbone'],state['cohort_path'])
                if key not in datasets:datasets[key]=TrackEventData(key[0],key[1])
                data=datasets[key]
                if settings.get('feature_plan'):data=CropEventData(data,settings['feature_plan'],settings['feature_variant'])
                if sha256(data.cohort_path)!=state['cohort_sha256']:raise ValueError('endpoint cohort changed')
                if len(data.fit)>256:
                    chosen={tuple(row['key']) for row in select_fit([data.items[i] for i in data.fit],256)}
                    indices=[i for i in data.fit if tuple(data.items[i]['key']) in chosen]
                else:indices=list(data.fit)
                model=build_model(settings);model.load_state_dict(state['model'],strict=True)
                fitted=evaluate(model,data,indices,'cpu')
                native=evaluate(None,data,indices,'cpu',scorer='native')
                diagnostic=dict(status='COMPLETE_RETAINED_FIT_ENDPOINT_DIAGNOSTIC',tag=tag,completed_updates=m['completed_steps'],
                    source_checkpoint_sha256=m['checkpoint_sha256'],cohort_sha256=state['cohort_sha256'],fit_queries=len(indices),
                    selected_fit_keys=[data.items[i]['key'] for i in indices],fit_metrics=fitted['metrics'],native_fit_metrics=native['metrics'],
                    retained_validation_metrics=m['validation_metrics'],fit_rows=fitted['rows'],new_model_updates=0,
                    selection='Up to256 original fit keys, scene/hash round robin independent of GT/error/teacher output. Crop variants use their same fixed256 fit inputs.',
                    limits='Retrospective hard-inference endpoint evaluation, not an online training-time Recall curve.150-step AG-C was trained with soft pooling but evaluated here with the declared hard inference sampler. Fit populations differ between cohorts; validation300 stays identical.',
                    provenance=provenance())
                save_json(target,diagnostic);reports.append(dict(tag=tag,path=str(target),sha256=sha256(target),metrics=diagnostic['fit_metrics']))
                print(json.dumps(dict(tag=tag,fit_Recall1=fitted['metrics']['Recall1'],validation_Recall1=m['validation_metrics']['Recall1'])),flush=True)
    summary=dict(status='SAVED_ENDPOINT_DIAGNOSTICS_COMPLETE' if not pending else 'PARTIAL_RETAINED_ENDPOINT_DIAGNOSTICS',
        completed=reports,pending=pending,model_updates=0,GT_not_used_for_fit_example_selection=True,provenance=provenance())
    save_json(ROOT/'results/trackvggt_fit_endpoint_diagnostics_v1.json',summary)


if __name__=='__main__':main()

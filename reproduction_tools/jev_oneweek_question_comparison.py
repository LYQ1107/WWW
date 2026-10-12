"""Equal-evidence typed-question comparison on all real differing known targets."""
import collections
import torch
from jev_oneweek_common import *
from jev_phase14_artifacts import load_dense
from jev_oneweek_assess import load_model
from jev_oneweek_learning import batch


def main():
    protect();arms=['jev','mlp','attention','deepsets','threshold']
    networks={a:load_model(a,20261009,'formal')[0] for a in arms}
    audit=read(REPORTS/'SUPERVISION_AUDIT.json')
    labels={tuple(r['key']):r for r in load_dense(audit['labels']['path'])['labels']}
    counts={a:collections.defaultdict(collections.Counter) for a in arms}
    for video in [12,13,14]:
        descriptor=read(PHASE15/'commitment_dataset_v2'/f'video{video:02d}/RESULT.json')['DATASET']
        records=load_dense(descriptor['path'])['records']
        for record in records:
            annotation=labels[tuple(record['key'])];mask=(annotation['cross']>=0)&(annotation['stability']>=0)&(annotation['cross']!=annotation['stability'])
            if not mask.any():continue
            x,y=batch([dict(record,oneweek_labels=annotation)]);q=len(record['rows']);k=len(record['refs']);mask=mask.to('cuda:0')
            group='new_gradient_TRAIN' if annotation['train_gradient'] else 'pre_exposed_temporal_reserved'
            for arm,model in networks.items():
                with torch.no_grad():details=model.details(x)
                a=details['cross_logits'][0,:q,:k]>=0;b=details['stability_logits'][0,:q,:k]>=0
                cross=y['cross'][0,:q,:k];stability=y['stability'][0,:q,:k];c=counts[arm][group]
                c['known_different_targets']+=int(mask.sum());c['cross_correct']+=int(((a==cross)&mask).sum())
                c['stability_correct']+=int(((b==stability)&mask).sum())
                c['both_questions_correct']+=int(((a==cross)&(b==stability)&mask).sum())
                c['predicted_questions_differ']+=int(((a!=b)&mask).sum())
    save(REPORTS/'TYPED_QUESTION_ORDINARY_CONTROL_COMPARISON.json',dict(status='COMPLETE',
         binding=binding('All known differing supervised question targets across authorized TRAIN'),
         methods={a:{g:dict(v) for g,v in groups.items()} for a,groups in counts.items()},
         same_inputs_supervision_update_budget_and_last_checkpoint=True,
         subset_selection_GT_OFFLINE_ONLY_not_deployment_event_selection=True,
         exposed_TRAIN_not_independent_scenes=True,no_future_labels=True,
         ordinary_two_output_heads_can_answer_two_different_questions=True))
    print('TYPED_CONTROL_COMPARISON_COMPLETE',{a:{g:dict(v) for g,v in c.items()} for a,c in counts.items()},flush=True)


if __name__=='__main__':main()

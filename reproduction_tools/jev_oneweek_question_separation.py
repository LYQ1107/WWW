"""Audit rare independently differing typed targets, preserving exposure strata."""
import collections
import torch
from jev_oneweek_common import *
from jev_phase14_artifacts import load_dense
from jev_oneweek_assess import load_model
from jev_oneweek_learning import batch
from gtr.modeling.jev_oneweek.plans import plans


def main():
    protect();model,checkpoint,_=load_model('jev',20261009,'formal')
    labels={tuple(r['key']):r for r in load_dense(read(REPORTS/'SUPERVISION_AUDIT.json')['labels']['path'])['labels']}
    strata=collections.defaultdict(collections.Counter);coverage=collections.Counter()
    with torch.no_grad():
        for video in TRAIN:
            records=load_dense(read(PHASE15/'commitment_dataset_v2'/f'video{video:02d}/RESULT.json')['DATASET']['path'])['records']
            for record in records:
                annotation=labels[tuple(record['key'])]
                a=annotation['cross'];b=annotation['stability'];different=(a>=0)&(b>=0)&(a!=b)
                if different.any():
                    x,y=batch([dict(record,oneweek_labels=annotation)]);d=model.details(x)
                    stratum='new_gradient_TRAIN12_13_14' if annotation['train_gradient'] else 'pre_exposed_temporal_reserved' if video!=16 else 'TRAIN16_pre_exposed_controller_audit'
                    c=strata[stratum];p=d['cross_logits'][0,:len(record['rows']),:len(record['refs'])]>=0
                    q=d['stability_logits'][0,:len(record['rows']),:len(record['refs'])]>=0
                    mask=different.to(p.device);aa=a.to(p.device);bb=b.to(p.device)
                    c['independently_different_edges']+=int(mask.sum());c['events']+=1
                    c['cross_correct']+=int(((p==aa)&mask).sum());c['stability_correct']+=int(((q==bb)&mask).sum())
                    c['both_questions_correct_on_differing_truth']+=int(((p==aa)&(q==bb)&mask).sum())
                    c['two_predicted_answers_differ']+=int(((p!=q)&mask).sum())
                if video==16:
                    # Native candidate coverage only under the fixed baseline
                    # scorer; certification includes private DEFER and UNKNOWN.
                    x,_=batch([dict(record,oneweek_labels=annotation)])
                    base=model.identity({k:v for k,v in x.items() if k!='commitment_features'})[0]
                    alternatives=plans(base,x['legal'][0]);coverage['payloads']+=1
                    known=record['known_options'];positive=record['positive'];k=len(record['refs'])
                    certificates=[]
                    for plan in alternatives:
                        chosen=torch.tensor([k if c<0 else c for c in plan]);rows=torch.arange(len(plan))
                        certified=bool(known[rows,chosen].all());correct=bool(positive[rows,chosen].all()) if certified else None
                        certificates.append(correct);coverage['proposals']+=1
                        coverage['UNKNOWN_complete_proposals']+=not certified
                    coverage['certified_correct_complete_proposal_exists']+=True in certificates
                    coverage['no_certified_correct_complete_proposal']+=True not in certificates
                    coverage['all_proposals_incompletely_certified']+=all(c is None for c in certificates)
                    coverage['certified_correct_base_complete_proposal']+=certificates[0] is True
            print('QUESTION_SEPARATION_VIDEO',video,{s:dict(c) for s,c in strata.items()},flush=True)
    save(REPORTS/'QUESTION_SEPARATION_AND_PLAN_COVERAGE.json',dict(status='COMPLETE',checkpoint=checkpoint,
         binding=binding('Observed typed-label disagreement with explicit training/exposure strata'),
         strata={s:dict(c) for s,c in strata.items()},TRAIN16_legal_proposal_coverage=dict(coverage),
         no_certified_correct_proposal_does_not_mean_no_physically_correct_proposal=True,
         UNKNOWN_complete_proposals_not_labelled_wrong=True,
         new_gradient_rows_not_independent_validation=True,pre_exposed_reserved_rows_not_new_independent_scenes=True,
         no_proposal_or_checkpoint_selected_using_this_analysis=True))


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        failure('question_separation',traceback.format_exc());raise

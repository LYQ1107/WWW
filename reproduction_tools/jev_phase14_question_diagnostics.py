"""Frozen same-input question/memory dependencies; no retraining advantage claim."""
import collections
import torch
from jev_phase14_common import *
from jev_phase13_learning import collate
from gtr.modeling.jev_stage2.model import GlobalIdentityJev
from gtr.modeling.jev_stage2.assignment import lawful_choice

class QuestionProbe(GlobalIdentityJev):
    def encode_questions(self,x,state,h,det):
        if self.probe=='unchanged':return super().encode_questions(x,state,h,det)
        if self.probe=='static_type':
            self.variant='fixed_question'
            try:return super().encode_questions(x,state,h,det)
            finally:self.variant='full'
        b,q,d=det.shape;k=h.shape[1];query=det+self.task(x['question_type'])
        weights=x['identity_mask'][...,None].float()
        bank=(h.sum(2)*weights).sum(1)/weights.sum(1).clamp_min(1) if k else det.new_zeros(b,d)
        own=(h[:,:,2]*weights).sum(1)/weights.sum(1).clamp_min(1) if k else bank
        other=(h[:,:,3]*weights).sum(1)/weights.sum(1).clamp_min(1) if k else bank
        ev=self.evidence(x['pair_evidence'])
        avg=(ev*x['legal'][...,None]).sum(2)/x['legal'].sum(2).clamp_min(1)[...,None] if k else det.new_zeros(b,q,d)
        tokens=torch.stack([query,self.detection(x['detection_meta']),bank[:,None].expand(-1,q,-1),own[:,None].expand(-1,q,-1),other[:,None].expand(-1,q,-1),avg],2)
        keep={'detection_conditioned':[0,1], 'availability_evidence':[0,5],
              'candidate_conflict':[0,2,5], 'cross_camera_evidence':[0,2,3,4]}[self.probe]
        remove=[i for i in range(6) if i not in keep];tokens[:,:,remove]=0
        flat=self.question_evidence(tokens.reshape(b*q,6,d),tokens.reshape(b*q,6,d))
        flat=self.question_read(flat,state[:,None].expand(-1,q,-1,-1).reshape(b*q,8,d))
        return flat.reshape(b,q,6,d),flat.mean(1).reshape(b,q,d)

def delete_factor(x,name):
    x={k:v.clone() for k,v in x.items()}
    for factor in name.split('+'):
        slots={'recent':0,'global_mean':1,'own_mean':2,'other_mean':3}
        if factor in slots:
            c=slots[factor];x['history_visual'][:,:,c]=0;x['history_mask'][:,:,c]=False;x['pair_evidence'][:,:,:,c]=0
        elif factor=='camera_meta':
            x['detection_meta'][:,:,5]=0;x['identity_meta'][:,:,[4,5,8]]=0;x['pair_evidence'][:,:,:,[5,6]]=0
        elif factor=='time':
            x['detection_meta'][:,:,6]=0;x['identity_meta'][:,:,[0,6,7,9]]=0;x['pair_evidence'][:,:,:,[4,14]]=0
        elif factor=='geometry':x['detection_meta'][:,:,:4]=0;x['pair_evidence'][:,:,:,10:15]=0
        elif factor=='counts':x['identity_meta'][:,:,[1,2,10]]=0;x['pair_evidence'][:,:,:,[7,15]]=0
        elif factor not in ['unchanged','shared_state']:raise ValueError(factor)
    return x

@torch.no_grad()
def main():
    protect();torch.set_num_threads(1);frames=read(REPORTS/'PREREGISTRATION.json')['diagnostics']['paired_prefix_frames']
    samples=[];data_manifest=[]
    for v in TRAIN:
        m=read(OLD/'dense_native_v3'/f'video{v:02d}'/'RESULT.json');assert sha(m['DATASET']['path'])==m['DATASET']['SHA256']
        records=torch.load(m['DATASET']['path'],map_location='cpu')['records']
        samples.extend(r for r in records if r['task']==0 and r['key'][1] in frames);data_manifest.append(m['DATASET'])
        del records
    probes=['unchanged','static_type','detection_conditioned','availability_evidence','candidate_conflict','cross_camera_evidence']
    factors=['unchanged','recent','global_mean','own_mean','other_mean','camera_meta','time','geometry','counts','shared_state',
             'own_mean+other_mean','global_mean+counts','other_mean+camera_meta']
    all_results=[];question_results=[]
    for seed in [20261008,20261009,20261010]:
        m=read(OLD/'formal_training_v1/full'/f'seed{seed}'/'RESULT.json');ck=m['checkpoint'];assert sha(ck['path'])==ck['SHA256']
        model=QuestionProbe().cuda().eval();model.load_state_dict(torch.load(ck['path'],map_location='cpu')['model']);model.probe='unchanged'
        reference={}
        for r in samples:
            x,_=collate([r]);z=model(x)[0];reference[tuple(r['key'])]=lawful_choice(z,x['legal'][0])
        for family,conditions in [('questions',probes),('memory',factors)]:
            for condition in conditions:
                counts=collections.Counter();changed=[]
                model.probe=condition if family=='questions' else 'unchanged'
                model.variant='no_shared_state' if condition=='shared_state' else 'full'
                for r in samples:
                    x,_=collate([r]);x=delete_factor(x,condition) if family=='memory' else x
                    z=model(x)[0];choices=lawful_choice(z,x['legal'][0]);k=len(r['refs']);old=reference[tuple(r['key'])]
                    for row,c in enumerate(choices):
                        selected=k if c<0 else c;counts['rows']+=1
                        if r['supervised'][row]:
                            counts['certified_rows']+=1
                            tag='UNKNOWN_selected' if not r['known_options'][row,selected] else 'certified_correct' if r['positive'][row,selected] else 'certified_wrong'
                            counts[tag]+=1
                        if c!=old[row]:
                            counts['assignment_changed']+=1
                            oldc=k if old[row]<0 else old[row]
                            if r['known_options'][row,oldc] and r['known_options'][row,selected]:
                                counts['certified_regression']+=int(r['positive'][row,oldc] and not r['positive'][row,selected])
                                counts['certified_correction']+=int(not r['positive'][row,oldc] and r['positive'][row,selected])
                            if len(changed)<12:changed.append({'key':r['key'],'row':row,'original':old[row],'intervened':c})
                item=dict(seed=seed,condition=condition,counts=dict(counts),examples=changed,checkpoint=ck)
                (question_results if family=='questions' else all_results).append(item)
                print('PHASE14_DEPENDENCY',family,seed,condition,dict(counts),flush=True)
        del model;torch.cuda.empty_cache()
    scope='same real B1 prefix causal input tensors and legal solver; out-of-distribution frozen inference dependency, not retrained structural ablation or mutated-future benefit'
    save(REPORTS/'DYNAMIC_QUESTION_DIAGNOSTICS.json',dict(status='COMPLETE',binding=binding(),scope=scope,results=question_results,
        original_task_supervision=['MATCH only'],old_Q2_Q3_learned=False,multiple_typed_questions='new pilot required; never claim old random task embeddings are three learned tasks',data=data_manifest,
        static_probe='frozen Full with task-only question tokens, separately from genuinely trained Fixed checkpoint'))
    save(REPORTS/'IDENTITY_MEMORY_FACTORIAL.json',dict(status='COMPLETE_DEPENDENCY_DIAGNOSTIC',binding=binding(),scope=scope,results=all_results,data=data_manifest,
        retained_information={'camera_meta':'own/other token roles remain; only explicit metadata removed','time':'motion-derived displacement still contains temporal consequences',
         'counts':'legal candidate cardinality remains visible through input shape; only numeric history counts removed'},retrained_structural_claim=False))
    save(REPORTS/'CROSS_CAMERA_EVIDENCE_AUDIT.json',dict(status='COMPLETE_DEPENDENCY_DIAGNOSTIC',binding=binding(),scope=scope,
        results=[x for x in all_results if x['condition'] in ['own_mean','other_mean','camera_meta','own_mean+other_mean','other_mean+camera_meta']],
        paired_actual_mutated_future='PENDING_NATIVE_VALIDATION',cross_camera_metric='requires actual official MATLAB; dependency decisions are not CVIDF1'))

if __name__=='__main__':main()

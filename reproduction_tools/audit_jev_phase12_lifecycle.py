"""Bounded genuine native WRITE/KEEP forks and observed stale-bank qualification."""
import argparse,time,types,collections
import torch
from jev_phase12_common import *
from gtr.modeling.jev_native_state import NativeProductionPrefixRecorder,NativeStateForkAdapter,fingerprint
from run_jev_phase10_closed_loop import raw_predictions

def run(video):
    protect();source=binding();assert not source['dirty'];assert video in TRAIN
    protocol=json.loads((REPORTS/'LIFECYCLE_AUDIT_PROTOCOL.json').read_text());selections=[e for e in protocol['events'] if e['key'][0]==video]
    manifest=json.loads((PREVIOUS/'native_capture_v1'/f'video{video:02d}/compat/RESULT.json').read_text());registered={tuple(e['key']):e for e in manifest['prefixes']}
    model=build_model(video);model.visual_jev_enabled=False;rows=inputs(video,manifest['frames']);out=OUT/'lifecycle_native_v1'/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True)
    from jev_phase7_offline import IdentityEvaluator
    from jev_phase8_opportunity import PrefixIdentityAnchors
    from jev_phase8_utility import effects
    evaluator=IdentityEvaluator(video);original_memory=model._jev_memory_action;original_react=model._jev_reactivation_action;original_bank=model.memory_bank;original_asso=model.get_asso;results=[]
    for selected in selections:
        key=tuple(selected['key']);entry=registered[key];target=selected['identity'];row=selected['row'];prefix=torch.load(entry['path'],map_location='cpu');anchors=PrefixIdentityAnchors()
        past=prefix['instances'][:key[1]*2+key[2]];aligned=evaluator.align(raw_predictions(past,evaluator.images_by_key if hasattr(evaluator,'images_by_key') else {(int(i['frame_id'])-1,int(i['view_id'])-1):i for i in evaluator.images.values()}))
        groups=collections.defaultdict(dict)
        for (f,v,r),a in sorted(aligned.items()):groups[f,v][r]=a
        for (f,v),group in sorted(groups.items()):anchors.update({r:a['id'] for r,a in group.items()},{r:a['gt'] for r,a in group.items()},f)
        branches={};target_gt=anchors.reliable().get(target);stop=min(manifest['frames']-1,key[1]+63)
        for tag in ['WRITE','KEEP']:
            eventout=out/f'F{key[1]}V{key[2]}R{row}'/tag;recorder=NativeProductionPrefixRecorder(model,[],eventout);armed=[True];bank=[False];reads=[];stale=[];scores=[];gallery_sha=[]
            def memory(**kw):
                current=(model._jev_context['frame'],model._jev_context['view'])
                if armed[0] and current==key[1:] and kw['detection_index']==row and kw['track_id']==target:
                    armed[0]=False;return 'WRITE_MEMORY' if tag=='WRITE' else 'SKIP_MEMORY'
                return original_memory(**kw)
            def react(**kw):
                action=original_react(**kw);stale.append({'key':[model._jev_context['frame'],model._jev_context['view']],'candidate_ids':kw.get('candidate_track_ids',[]),'chosen_identity':kw.get('track_id'),'scores':kw.get('candidate_scores',[]),'action':action,'anchor_GT':{str(t):anchors.reliable().get(t) for t in kw.get('candidate_track_ids',[])}});return action
            def bank_call(*a,**kw):
                bank[0]=True
                try:return original_bank(*a,**kw)
                finally:bank[0]=False
            def asso(instances,*a,**kw):
                if bank[0]:
                    history=[i[i.track_ids==target].reid_features for i in instances[:-1] if i.has('track_ids') and (i.track_ids==target).any()]
                    if history:reads.append({'key':[model._jev_context['frame'],model._jev_context['view']],'target_vectors_SHA256':fingerprint(torch.cat(history)),'actual_bank_prototype_READ':True})
                return original_asso(instances,*a,**kw)
            def after(**kw):
                packet=kw['candidate']
                if packet is not None:
                    b=packet['batch'];scores.append({'key':[kw['frame'],kw['view']],'candidate_ids':b.candidate_ids,'GMT_scores':b.scores.cpu().clone(),'evidence12':b.evidence12.cpu().clone()})
                if (kw['frame'],kw['view'])==key[1:]:gallery_sha.append(fingerprint(kw['galleries'][target]))
                recorder.after(**kw)
            model._jev_memory_action=memory;model._jev_reactivation_action=react;model.memory_bank=bank_call;model.get_asso=asso;model.jev_native_prefix_observer=recorder.before;model.jev_candidate_commit_observer=after
            with torch.no_grad():instances,_=NativeStateForkAdapter(model).run(entry['path'],rows,stop_frame=stop)
            assert not armed[0],'selected genuine WRITE event was not reached'
            prediction=raw_predictions(instances,{(int(i['frame_id'])-1,int(i['view_id'])-1):i for i in evaluator.images.values()});observed=evaluator.align(prediction)
            branches[tag]={'native_trace':recorder.trace,'bank_reads':reads,'stale_questions':stale,'target_gallery_SHA256':gallery_sha[0],'effects':{str(h):effects(observed,anchors.diagnostics(),key[1],h,{target_gt} if target_gt is not None else set()) for h in [32,64]},'score_packets':scores}
        a,b=branches['WRITE'],branches['KEEP'];firsta,firstb=a['native_trace'][0],b['native_trace'][0]
        assert firsta['ids']==firstb['ids'] and firsta['events'][row]['gallery_after']==firstb['events'][row]['gallery_after']+1
        changed_ids=[x['key'] for x,y in zip(a['native_trace'],b['native_trace']) if x['ids']!=y['ids']];readchanged=a['bank_reads']!=b['bank_reads'];scorechanged=0;evidencechanged=0
        for x,y in zip(a['score_packets'],b['score_packets']):
            same=x['candidate_ids']==y['candidate_ids'] and x['GMT_scores'].shape==y['GMT_scores'].shape
            scorechanged+=not same or not torch.equal(x['GMT_scores'],y['GMT_scores']);evidencechanged+=not same or not torch.equal(x['evidence12'],y['evidence12'])
        delta={h:a['effects'][h]['utility']-b['effects'][h]['utility'] for h in ['32','64']}
        for branch in branches.values():branch.pop('score_packets')
        result={'key':key,'row':row,'identity':target,'prefix_SHA256':entry['sha256'],'reliable_prefix_anchor':target_gt,'Gallery_changed':a['target_gallery_SHA256']!=b['target_gallery_SHA256'],'actual_bank_READ_count':len(a['bank_reads']),'bank_READ_changed':readchanged,'next_native_score_packets_changed':scorechanged,'numerical_evidence_packets_changed':evidencechanged,'committed_ID_changed_keys':changed_ids,'utility_WRITE_minus_KEEP':delta,'non_tie_H64':delta['64']!=0,'branches':branches}
        save(out/f'EVENT_{key[1]}_{key[2]}_{row}.json',result);results.append(result);save(out/'PROGRESS.json',{'status':'RUNNING','completed':len(results),'total':len(selections)});print('NATIVE_WRITE_KEEP_H64',video,key,delta,flush=True)
    save(out/'RESULT.json',{'status':'COMPLETE','binding':source,'events':results,'scope':'8 preregistered genuine TRAIN-native WRITE/KEEP interventions, GMT_OFF frozen future; bounded identifiability audit, not exhaustive absence proof','heldout':'SEALED'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);run(p.parse_args().video)

"""Audit every frozen B2 REASSOCIATE and isolated live-policy replacements."""
import copy
import json
import torch

from jev_phase6_common import ROOT,OUT,REPORTS,sha,save,protect_anchor
from jev_phase6_rollouts import NativeReplayLab,event_key
from jev_phase6_offline_utility import OfflineIdentityAudit,whole_run_correct,prefix_identity,consequence


def main(device):
    lab=NativeReplayLab(1,device); captures=[]
    def collect(feature,question,legal,action,context):
        if question=='MATCH_DECISION' and action=='REASSOCIATE':
            captures.append({'key':(1,context['frame'],context['view'],question,context['detection_index']),
                             'state':lab.prefix.clone(),'context':copy.deepcopy(context),
                             'feature':feature.detach().cpu().tolist(),'legal':list(legal)})
        return action
    result=lab.run(OUT/'reassociate/baseline',intervention=collect)
    reference=json.loads((OUT/'gating_tracking/G5/result.json').read_text())
    if sha(result['predictions'])!=reference['predictions_sha256'] or len(captures)!=11:
        raise AssertionError('instrumented B2 does not reproduce all 11 frozen events')
    gt=OfflineIdentityAudit(1)
    actual=gt.align(json.loads(open(result['predictions']).read()))
    gmt_result=json.loads((OUT/'gating_tracking/G0/result.json').read_text())
    gmt=gt.align(json.loads(open(gmt_result['result']['predictions']).read()))
    ca=whole_run_correct(actual); co=whole_run_correct(gmt); events=[]
    for index,c in enumerate(captures):
        key=c['key']; local=(key[1],key[2],key[4]); start=(key[1],key[2]); target=actual[local]['gt']
        prefix_map,purity=prefix_identity(actual,start)
        snapshots=OUT/'reassociate/snapshots';snapshots.mkdir(parents=True,exist_ok=True)
        torch.save(c,snapshots/f'{index:02d}.pth')
        branches={}; fork_rows={}
        for label,forced in [('CONTROL',None),('FORCE_NEW','START_NEW'),('FORCE_ACCEPT','ACCEPT_CURRENT')]:
            def intervene(feature,question,legal,action,context):
                return forced if forced is not None and (int(context['video_id']),int(context['frame']),int(context['view']),question,int(context['detection_index']))==key else action
            r=lab.run(OUT/f'reassociate/forks/{index:02d}/{label}',prefix=c['state'],start_key=key[:3],
                      end_frame=key[1]+32,intervention=intervene)
            aligned=gt.align(json.loads(open(r['predictions']).read())); fork_rows[label]=aligned
            if label=='CONTROL':
                expected={k:v for k,v in actual.items() if k[:2]>=start and k[0]<=start[0]+32}
                if aligned!=expected: raise AssertionError(f'fork control state/RNG mismatch at {key}')
            branches[label]={'predictions_sha256':sha(r['predictions']),'rows':len(aligned),
                'future':{str(h):consequence(aligned,actual,start,target,h,c['state'].next_id) for h in (1,2,4,8,16,32)}}
        entry={'index':index,'frame':key[1],'view':key[2],'row':key[4],
               'proposal_track':c['context']['proposal_track_id'],'second_best_candidate':c['context']['alternate_track_id'],
               'candidate_ids':c['context']['candidate_track_ids'],'candidate_scores':c['context']['candidate_scores'],
               'B2_action':'REASSOCIATE','GMT_action':next((d['action'] for d in json.loads(open(gmt_result['result']['decisions']).read()) if d['question']=='MATCH_DECISION' and (d['frame'],d['view'],d['row'])==local),None),
               'actual_committed_id':actual[local]['id'],'GT':target,'prefix_GT_by_id':prefix_map,'prefix_GT_counts':purity,
               'N01':int(not co[local] and ca[local]),'N10':int(co[local] and not ca[local]),
               'branches':branches,'control_prefix_and_rng_exact':True,
               'snapshot_sha256':sha(snapshots/f'{index:02d}.pth')}
        events.append(entry)
        save(OUT/'reassociate/progress.json',{'completed':len(events),'total':11})
    totals={'N01':sum(e['N01'] for e in events),'N10':sum(e['N10'] for e in events)}
    for branch in ('FORCE_NEW','FORCE_ACCEPT'):
        totals['control_better_than_'+branch]=sum(e['branches']['CONTROL']['future']['32']['utility']>e['branches'][branch]['future']['32']['utility'] for e in events)
        totals['control_worse_than_'+branch]=sum(e['branches']['CONTROL']['future']['32']['utility']<e['branches'][branch]['future']['32']['utility'] for e in events)
    save(REPORTS/'REASSOCIATE_EVENT_AUDIT.json',{'status':'COMPLETE','events':events,'totals':totals,
        'GT_only_offline':True,'coordinate_contract':'cache0_annotation1, exact annotation image IDs; IoU>=0.5 one-to-one',
        'N01_N10_contract':'per-camera whole-run ID mapping, observational GMT/B2 comparison; isolated fork utility uses fixed pre-event identity mapping',
        'future_policy':'B2 MATCH, instantaneous GMT MEMORY/REACTIVATION; branch state and RNG recomputed',
        'overlapping_windows_not_independent':True,'what_did_we_learn':'Use each isolated intervention and the complete G4 replacements together; event rarity alone does not establish necessity.',
        'official_test_read':False,'full24_authorized':False})
    print(json.dumps({'status':'COMPLETE','events':len(events),'totals':totals}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');main(p.parse_args().device)

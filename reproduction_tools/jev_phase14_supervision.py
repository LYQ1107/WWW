"""Certify Q2/Q3 from real saved B1 commits reconstructed from saved scores."""
import collections
import gc
import time
import numpy as np
import torch
from jev_phase14_common import *
from jev_phase14_forensics import native_choice

def label_video(video):
    from jev_phase13_runtime import cache_inputs
    from build_dense_jev_stage2_dataset import OfflineLabels
    source = read(OLD/'dense_native_v3'/f'video{video:02d}'/'RESULT.json')
    data = source['DATASET']; assert sha(data['path'])==data['SHA256']
    records = torch.load(data['path'], map_location='cpu')['records']
    groups = collections.OrderedDict()
    for i, r in enumerate(records): groups.setdefault(tuple(r['key']), []).append((i, r))
    first = next(iter(groups.values()))[0][1]; first_view = first['key'][2]
    _, _, reader = cache_inputs(video); labels = OfflineLabels(video, reader)
    boot = labels.current(0, 1-first_view)
    votes = collections.defaultdict(collections.Counter); total = collections.Counter()
    for ref, gt in enumerate(boot, 1):
        total[ref] += 1
        if gt is not None: votes[ref][gt] += 1
    id_count = len(boot); count = collections.Counter(); output=[]; paired=[]
    unique_trust = collections.defaultdict(set)
    for key, items in groups.items():
        primary = next(r for _,r in items if r['task']==0)
        current_GT = primary['GT_labels_OFFLINE_ONLY']; ids=[-1]*len(primary['rows'])
        for index, r in items:
            refs=r['refs']; k=len(refs); pos=r['positive']; known=r['known_options']
            target=r['GT_labels_OFFLINE_ONLY']; avail=torch.full((len(target),),-1,dtype=torch.int8)
            trust=torch.full((k,),-1,dtype=torch.int8)
            expected_known=torch.tensor([len(votes[t])==1 for t in refs],dtype=torch.bool)
            for c,t in enumerate(refs):
                coverage=sum(votes[t].values())/max(1,total[t])
                if sum(votes[t].values())>=3 and coverage>=.8:
                    if len(votes[t])==1: trust[c]=1
                    elif sum(n>=2 for n in votes[t].values())>=2: trust[c]=0
                if trust[c]>=0: unique_trust[int(trust[c])].add((t,key[1]//64));count['Q3_pure' if trust[c]==1 else 'Q3_contaminated']+=1
            withheld_rows=[]
            for row,gt in enumerate(target):
                if gt is None:count['current_GT_UNKNOWN']+=1;continue
                # Reconstructed support must exactly reproduce original saved labels.
                assert torch.equal(known[row,:k], expected_known), (video,key,row,'known')
                expected_positive=torch.tensor([set(votes[t])=={gt} for t in refs],dtype=torch.bool)
                assert torch.equal(pos[row,:k],expected_positive),(video,key,row,'positive')
                if expected_positive.any():avail[row]=1;count['Q2_natural_positive']+=1
                elif expected_known.all():avail[row]=0;count['Q2_natural_negative']+=1
                else:count['Q2_natural_UNKNOWN']+=1
                # Clean candidate presence alone cannot certify the remaining UNKNOWN identities absent.
                if expected_positive.any() and expected_known.all():
                    withheld_rows.append(row);count['Q2_paired_withholding_eligible']+=1
                    if k>=20:count['many_candidate_withholding']+=1
                    if bool(r['inputs']['history_mask'][0,expected_positive,3].any()):count['cross_camera_withholding']+=1
            audit_block=(key[1]//64)%5==4
            output.append(dict(source_record_index=index,key=r['key'],task=r['task'],availability=avail,
                               trust=trust,withholdable_rows=withheld_rows,audit_block=audit_block,
                               provenance='past observed GT support only; neural inputs unchanged'))
            if withheld_rows and r['task']==0:
                paired.append(dict(source_record_index=index,rows=withheld_rows,key=r['key'],audit_block=audit_block))
            values=r['inputs']['pair_evidence'][0,:,:,[0,1,2]].max(-1).values
            z=torch.cat([values,values.new_full((len(values),1),.75)],-1)
            choices=native_choice(z.numpy(),r['inputs']['legal'][0].numpy())
            for row,c in zip(r['rows'],choices):
                if c>=0:
                    assert ids[row]<0
                    ids[row]=refs[c]
        for row,gt in enumerate(current_GT):
            if ids[row]<0:id_count+=1;ids[row]=id_count
            total[ids[row]]+=1
            if gt is not None:votes[ids[row]][gt]+=1
    assert id_count==source['natural_native_ids'],(video,id_count,source['natural_native_ids'])
    count['Q3_unique_pure_identity_time_blocks']=len(unique_trust[1])
    count['Q3_unique_contaminated_identity_time_blocks']=len(unique_trust[0])
    out=OUT/'supervision_v1'/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(labels=output,paired=paired,source_dataset=data,video=video,GT_actor_inputs=False),out/'LABELS.pth')
    result=dict(status='COMPLETE',video=video,counts=dict(count),source_DATASET=data,
                label_artifact=dict(path=str(out/'LABELS.pth'),SHA256=sha(out/'LABELS.pth')),
                exact_original_candidate_label_reconstruction=True,native_final_ID_count=id_count,
                Q3_purity_scope='observed known GT coverage >=0.8, >=3 known observations; no majority vote on mixtures',
                natural_absence_requires_all_lawful_options_certified=True)
    save(out/'RESULT.json',result);del records,labels;gc.collect()
    print('PHASE14_Q2_Q3_VIDEO',video,dict(count),flush=True)
    return result

def main():
    protect();start=time.monotonic();torch.set_num_threads(1);results=[label_video(v) for v in TRAIN]
    counts=collections.Counter()
    for r in results:counts.update(r['counts'])
    criteria=read(REPORTS/'PREREGISTRATION.json')['labels']
    natural=counts['Q2_natural_positive']>=criteria['natural_Q2_qualification']['positive_rows_min'] and counts['Q2_natural_negative']>=criteria['natural_Q2_qualification']['negative_rows_min'] and sum(r['counts'].get('Q2_natural_negative',0)>0 for r in results)>=2
    intervened=counts['Q2_paired_withholding_eligible']>=500 and sum(r['counts'].get('Q2_paired_withholding_eligible',0)>0 for r in results)>=3
    trust=counts['Q3_pure']>=100 and counts['Q3_contaminated']>=50 and sum(r['counts'].get('Q3_contaminated',0)>0 for r in results)>=2
    save(REPORTS/'QUESTION_SUPERVISION_ELIGIBILITY.json',dict(status='PASS' if (natural or intervened) and trust else 'QUALIFICATION_FAILED',binding=binding(),
        counts=dict(counts),natural_Q2_qualified=natural,intervened_Q2_qualified=intervened,Q3_observed_history_trust_qualified=trust,
        videos=results,source='existing natural B1 native TRAIN12/13/14/16, no development GT used',
        lifetime_heads='REACT/MEMORY action heads remain NOT_RUN, Q3 purity supervision is not WRITE/KEEP reward',seconds=time.monotonic()-start))
    save(REPORTS/'AVAILABILITY_INTERVENTION_DATA.json',dict(status='PASS' if intervened else 'INSUFFICIENT_SUPPORT',binding=binding(),
        qualified_paired_rows=counts['Q2_paired_withholding_eligible'],videos=results,
        paired_control='unchanged natural source payload; TRAIN-only intervention physically removes all certified target options in a versioned input copy; remaining UNKNOWN options disqualify certified absence',
        legitimate_action='DEFER in MATCH, never natural NEW or stale recovery',
        difficulty='cosine competitor evidence: easy<0.5,hard>=0.5; manyK>=20; crosscamera target other-view present; ambiguous rows excluded from forced negatives',
        labels_not_actor_inputs=True))
    print('PHASE14_QUESTION_QUALIFICATION',natural,intervened,trust,dict(counts),flush=True)

if __name__=='__main__':main()

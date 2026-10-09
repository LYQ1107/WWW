"""Freeze natural supervision, candidate recall and TRAIN-only Tiny selection."""
import collections,gc
import numpy as np
import torch
from PIL import Image,ImageOps
from jev_phase13_learning import *

def main():
    protect();torch.set_num_threads(1);manifests=[];stats={};selected={};assessment=[];totals=collections.Counter();recalls={};ann=json.loads(ANNOTATIONS.read_text());videos={v['id']:v for v in ann['videos']};images={v:[i for i in ann['images'] if i['video_id']==v] for v in TRAIN+VAL};image_probes=[];numeric_labels={}
    for v in TRAIN+VAL:
        rows,mm=load_data([v]);m=mm[0];manifests.append(m);old=torch.load(OUT/'dense_native_v1'/f'video{v:02d}'/'DATASET.pth',map_location='cpu')['records'];assert len(rows)==len(old)
        ks=[];count=collections.Counter();support=collections.Counter();ranks=[];missing=collections.Counter();changed=0
        for r,o in zip(rows,old):
            assert r['key']==o['key'] and r['refs']==o['refs'] and r['rows']==o['rows'] and r['task']==o['task'];assert all(torch.equal(r['inputs'][k],o['inputs'][k]) for k in r['inputs'])
            changed+=int((r['supervised']!=o['supervised']).sum());q=len(r['rows']);k=len(r['refs']);ks.append(k);count['all_natural_rows']+=q;count['known_rows']+=int(r['supervised'].sum());count['UNKNOWN_options']+=int((~r['known_options'][:,:k]).sum());count.update(r['label_reasons'])
            assert not (r['positive'] & ~r['known_options']).any()
            for row in range(q):
                if r['GT_labels_OFFLINE_ONLY'][row] is None:continue
                scope='active' if r['task']==0 else 'stale';info=r['support_OFFLINE_ONLY'][row];positive=r['positive'][row,:k]
                support[scope+'_all_aligned_rows']+=1;support[scope+'_supported_rows']+=int(positive.any());support[scope+'_clean_global_available']+=info['available_clean_global'];support[scope+'_clean_global_in_pool']+=int(info['available_clean_global'] and positive.any())
                if positive.any():
                    sim=r['inputs']['pair_evidence'][0,row,:k,:3].max(-1).values;order=torch.argsort(sim,descending=True,stable=True);rank=int(torch.where(positive[order])[0][0])+1;ranks.append((scope,rank));count['cross_camera_positive_rows']+=int((r['inputs']['history_mask'][0,positive,3]).any())
                else:missing[scope+':'+r['label_reasons'][row]]+=1
            if v in TRAIN:
                key=tuple(r['key'])+(r['task'],);normal=r['task']==0 and (r['label_reasons'].count('CLEAN_ACTIVE_ID')>0)
                hard=False
                for rr in range(q):
                    pp=r['positive'][rr,:k]
                    if pp.any():
                        ss=r['inputs']['pair_evidence'][0,rr,:k,:3].max(-1).values;hard|=bool(ss.masked_fill(~pp,-1e4).max()<ss.max())
                conditions={'normal_video%02d'%v:normal and r['key'][1]>=40 and bool(r['inputs']['history_mask'][:,:,2].any()),'cross_camera':normal and bool(r['inputs']['history_mask'][:,:,3].any()),'competition':normal and k>=3 and bool((r['known_options'][:,:k].sum(1)>=3).any()),'hard_positive':normal and hard,'stale':r['task']==1 and 'CLEAN_STALE_ID' in r['label_reasons'],'genuine_scene_NEW':r['task']==1 and r['key'][1]>0 and any(reason=='NEW_TO_OBSERVED_MEMORY' and info['first_scene_GT_frame']==r['key'][1] for reason,info in zip(r['label_reasons'],r['support_OFFLINE_ONLY'])),'empty_candidate':k==0 and bool(r['supervised'].any())}
                for name,condition in conditions.items():
                    if condition and name not in selected:selected[name]={'record_key':list(key),'natural_rows':q,'legal_candidates':k,'label_reasons':r['label_reasons']}
            elif r['task']==0 and r['key'][1]%32==0:assessment.append(r['key']+[r['task']])
        summary={'groups':len(rows),'counts':dict(count),'candidate_count':dict(zip(['min','p25','p50','p75','p95','max'],map(float,np.quantile(ks,[0,.25,.5,.75,.95,1])))),'v1_v2_all_actor_inputs_and_candidate_refs_exact':True,'changed_label_rows_to_UNKNOWN':changed};stats[str(v)]=summary
        recall={'counts':dict(support),'all_aligned_support_rate':{s:support[s+'_supported_rows']/support[s+'_all_aligned_rows'] if support[s+'_all_aligned_rows'] else None for s in ['active','stale']},'clean_global_available_recall':{s:support[s+'_clean_global_in_pool']/support[s+'_clean_global_available'] if support[s+'_clean_global_available'] else None for s in ['active','stale']},'appearance_rank_recall_given_clean_positive_in_pool':{s:{str(K):sum(rank<=K for scope,rank in ranks if scope==s)/sum(scope==s for scope,rank in ranks) if any(scope==s for scope,rank in ranks) else None for K in [1,5,10,20,40,100]} for s in ['active','stale']},'missing_reasons':dict(missing)};recalls[str(v)]=recall
        if v in TRAIN:totals.update(count)
        ids={i['id'] for i in images[v]};numeric_labels[v]=set(a['instance_id'] for a in ann['annotations'] if a['image_id'] in ids)
        for view in [1,2]:
            iv=sorted([i for i in images[v] if i['view_id']==view],key=lambda i:i['frame_id'])
            for im in [iv[0],iv[len(iv)//2],iv[-1]]:
                path=IMAGES/im['file_name'];assert path.exists();image=Image.open(path);small=np.array(ImageOps.grayscale(image).resize((9,8)),dtype=np.int16);bits=(small[:,1:]>small[:,:-1]).ravel();dh=sum(int(bit)<<i for i,bit in enumerate(bits));image_probes.append({'video':v,'view':view,'frame':im['frame_id'],'path':str(path),'SHA256':sha(path),'dHash':hex(dh)})
        del rows,old;gc.collect()
    near=[]
    for a in image_probes:
        if a['video'] not in TRAIN:continue
        for b in image_probes:
            if b['video'] not in VAL:continue
            distance=bin(int(a['dHash'],16)^int(b['dHash'],16)).count('1')
            if distance<=5:near.append({'TRAIN':[a['video'],a['view'],a['frame']],'VAL':[b['video'],b['view'],b['frame']],'dHash_distance':distance,'exact_bytes':a['SHA256']==b['SHA256']})
    required=['normal_video12','normal_video13','normal_video14','normal_video16','cross_camera','competition','stale','genuine_scene_NEW','empty_candidate'];assert all(n in selected for n in required),selected
    dataset={'status':'PASS','binding':binding(),'version':'dense_native_v2','videos':manifests,'video_statistics':stats,'TRAIN_totals':dict(totals),'natural_all_payloads_no_corrective_filter':True,'no_GT_actor_tensors':True,'split':{'TRAIN':TRAIN,'development_VAL':VAL,'heldout':'20/21/22 SEALED'},'historical_Stage1_pretraining_exposure':'all 24 TRAIN videos including VAL and SEALED were used historically; development comparison only, no independent full-system generalization claim','REACT_qualification':{'status':'UNTRAINED_FALLBACK','real_stale_positive_rows':totals['CLEAN_STALE_ID'],'minimum_training_stale_rows':100,'minimum_training_videos_with_stale':3,'reason':'7 true stale positives are insufficient; no fake candidates/labels; untrained head disabled online'},'MEMORY_qualification':{'status':'UNTRAINED_FALLBACK','WRITE_KEEP_labels':0,'reason':'native WRITE fallback; no future-GMT H32 labels'},'tiny_frozen_selection':selected,'assessment_record_keys':assessment,'annotations_SHA256':sha(ANNOTATIONS)}
    save(REPORTS/'DENSE_DATASET_MANIFEST.json',dataset)
    save(REPORTS/'LABEL_AUDIT.json',{'status':'PASS','binding':binding(),'version':'dense_native_v2','video_statistics':stats,'GT_IoU_alignment':{'algorithm':'one-to-one Hungarian IoU>=0.5 to current official TRAIN annotation only','ambiguous_history':'UNKNOWN forever, never majority vote; potential same-target contaminated candidate makes terminal supervision UNKNOWN','unknown_option_loss':'excluded from conditional normalizer and structural negative choices; actual inference still admits lawful UNKNOWN candidates','DEFER':'no certified active candidate; not a claim target identity never existed','START_NEW':'new to observed identity memory; scene-first-frame flag reported separately'},'v1_correction':'v1 raw data preserved; v2 exactly same all actor tensors/state/refs, uncertain terminal labels excluded','identity_scope':'video-local GT numeric labels are not biometric global classes and never NN input'})
    save(REPORTS/'CANDIDATE_RECALL.json',{'status':'PASS','binding':binding(),'candidate_protocol':'ALL actual legal active/stale native identities, no GTA pruning/no TopK; private terminal','by_video':recalls,'definitions':'all-aligned support includes true-new/missing/contaminated cases; conditional rank recall only when a certified target exists; clean-global-available recall also measures frozen native window/bank availability, not merely retrieval','NEW_support':{str(m['video']):{'new_to_memory':m['counts'].get('genuine_NEW_labels',0),'scene_first_frame':m['counts'].get('NEW_scene_first_frame',0)} for m in manifests}})
    save(REPORTS/'SPLIT_LEAKAGE_AUDIT.json',{'status':'DEVELOPMENT_ONLY_PRETRAIN_EXPOSED','videos':{str(v):videos[v] for v in TRAIN+VAL},'numeric_GT_label_overlaps':{str(v):{str(w):len(numeric_labels[v]&numeric_labels[w]) for w in VAL} for v in TRAIN},'identity_overlap_interpretation':'GT numbers reset by sequence and are not proof of the same person; visual biometric disjointness cannot be certified from these labels','image_probes':image_probes,'near_duplicate_probes':near,'probe_scope':'first/middle/last frame per allowed camera; no exhaustive perceptual duplicate claim; no heldout pixels read','historical_feature_pretraining':'known exposure to all TRAIN24; heldout G9 remains NO_GO'})
    docs=ROOT/'docs/JEV_PHASE13_DENSE_DATASET.md';docs.write_text('# Phase XIII dense association data\n\nAll 12/13/14/16 TRAIN and 17/18/19 development payloads use freshly computed, unjittered Stage1 VFCE1024. Real B1 commits produce the history; no GTA scores, teacher forcing, future labels, or corrective filtering. All actual MATCH and queried REACT groups remain present.\n\nA detection is aligned to current TRAIN GT by one-to-one Hungarian IoU≥0.5. A historical predicted ID with multiple GT labels remains UNKNOWN. Potentially correct contaminated IDs do not authorize a DEFER label. Unknown options never become training negatives. Loss/calibration normalizers cover certified options only; full online inference still admits all legal candidates. START_NEW labels mean new to observed memory; scene-new counts are separate.\n\nVersion v1 is preserved. v2 changes uncertain labels and fixes the report field that mistakenly read `view_num` as `id_count`; every actor tensor, reference, and event ordering is unchanged. Per-video SHA, natural distributions, candidate support denominators, and frozen Tiny examples are in the linked JSON reports.\n\nThere are '+str(totals['CLEAN_ACTIVE_ID'])+' ordinary certified active positive rows in TRAIN, '+str(totals['CLEAN_STALE_ID'])+' stale recovery positives, and '+str(totals['NEW_TO_OBSERVED_MEMORY'])+' new-to-memory positives. REACT is unqualified for formal learning; MEMORY has no WRITE/KEEP supervision. Neither head may mutate native state. Their frozen fallbacks are shared by every formal method.\n\nStage1 historical pretraining included all 24 TRAIN sequences. Development scenes are wood/park, TRAIN scenes path/football, but numeric IDs alone cannot certify person disjointness. No independent full-system test claim is possible with inherited Stage1. video20/21/22 remain sealed. Sample near-duplicate checks do not establish exhaustive absence.\n')
    print('DENSE_PHASE13_AUDIT_PASS',dict(totals),flush=True)
if __name__=='__main__':main()

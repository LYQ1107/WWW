"""GT-only offline metrics. Never imported by live actors or transition bridge."""
from collections import Counter,defaultdict
import gzip,json,time
import numpy as np
from scipy.optimize import linear_sum_assignment
from jev_phase7_common import ANNOTATIONS,save
from audit_jev_phase5_decisions import xyxy,overlaps

class IdentityEvaluator:
    def __init__(self,video):
        ann=json.loads(ANNOTATIONS.read_text())
        self.images={i['id']:i for i in ann['images']if int(i['video_id'])==video}
        self.gt=defaultdict(list);self.denominator=Counter();self.gt_views=defaultdict(set)
        for a in ann['annotations']:
            if a['image_id']in self.images and a.get('conf',1)>0 and a.get('category_id',1)==1:
                self.gt[a['image_id']].append(a);g=int(a['instance_id']);self.denominator[g]+=1
                self.gt_views[g].add(int(self.images[a['image_id']]['view_id'])-1)
    def align(self,predictions):
        groups=defaultdict(list)
        for p in predictions:groups[p['image_id']].append(p)
        rows={}
        for image_id,dets in groups.items():
            image=self.images[image_id];gt=self.gt[image_id];matched={}
            if dets and gt:
                ious=overlaps(np.array([xyxy(p['bbox'])for p in dets]),np.array([xyxy(a['bbox'])for a in gt]))
                rr,cc=linear_sum_assignment(-((ious>=.5)*1000+ious))
                matched={int(r):int(gt[c]['instance_id'])for r,c in zip(rr,cc)if ious[r,c]>=.5}
            for row,p in enumerate(dets):
                rows[(int(image['frame_id'])-1,int(image['view_id'])-1,row)]={
                    'id':int(p['track_id']),'gt':matched.get(row),'bbox':p['bbox'],'score':p['score']}
        return rows
    @staticmethod
    def mapping(rows):
        counts=Counter((r['id'],r['gt'])for r in rows.values()if r['gt']is not None)
        ids=sorted({t for t,g in counts});gts=sorted({g for t,g in counts})
        if not ids or not gts:return {},counts
        matrix=np.array([[counts[(t,g)]for g in gts]for t in ids]);rr,cc=linear_sum_assignment(-matrix)
        return {ids[r]:gts[c]for r,c in zip(rr,cc)if matrix[r,c]>0},counts
    def summarize(self,rows):
        mapping,counts=self.mapping(rows);born={};known=0;global_wrong=0;unmapped=0;wrong=Counter();timelines=defaultdict(dict)
        ids_by_gt=defaultdict(set);gts_by_id=defaultdict(set);retrieval=defaultdict(list)
        for key,r in sorted(rows.items()):
            retrieval[r['id']].append((key,r));g=r['gt']
            if g is None:continue
            known+=1;born.setdefault(r['id'],g);ids_by_gt[g].add(r['id']);gts_by_id[r['id']].add(g)
            unmapped+=r['id']not in mapping;global_wrong+=mapping.get(r['id'])!=g
            bad=born[r['id']]!=g;wrong['birth_anchor_wrong_observations']+=bad
            timelines[(g,key[1])][key[0]]=bool(bad or timelines[(g,key[1])].get(key[0],False))
        episodes=[]
        for (g,view),timeline in sorted(timelines.items()):
            active=[];previous=None
            for frame,bad in sorted(timeline.items()):
                if active and (not bad or frame!=previous+1):
                    episodes.append({'gt':g,'view':view,'start':active[0],'end':active[-1],
                                     'duration_frames':len(active),'right_censored':frame!=previous+1});active=[]
                if bad:active.append(frame)
                previous=frame
            if active:episodes.append({'gt':g,'view':view,'start':active[0],'end':active[-1],
                                      'duration_frames':len(active),'right_censored':True})
        query=[];query_times=[];reverse={g:t for t,g in mapping.items()}
        for g,den in sorted(self.denominator.items()):
            stamp=time.perf_counter_ns();hits=retrieval.get(reverse.get(g),[]);query_times.append(time.perf_counter_ns()-stamp)
            tp=sum(r['gt']==g for k,r in hits);fp=len(hits)-tp;views={k[1]for k,r in hits if r['gt']==g}
            query.append({'gt':g,'predicted_id':reverse.get(g),'retrieved':len(hits),'tp':tp,'false_joins':fp,
                          'gt_observations':den,'precision':tp/len(hits)if hits else None,'recall':tp/den,
                          'cross_camera_continuity':len(views)>1,'expected_cross_camera':len(self.gt_views[g])>1})
        tp=sum(q['tp']for q in query);retrieved=sum(q['retrieved']for q in query);den=sum(self.denominator.values())
        durations=[e['duration_frames']for e in episodes]
        return {'definition':'IoU>=.5 one-to-one per image; fixed global two-camera identity map; separate first-observation anchor temporal errors',
            'known_observations':known,'unknown_GT_observations':len(rows)-known,'global_identity_wrong_observations':global_wrong,
            'global_unmapped_fragment_observations':unmapped,'birth_anchor_wrong_observations':wrong['birth_anchor_wrong_observations'],
            'wrong_ID_episodes':episodes,'wrong_ID_duration_total_camera_frames':sum(durations),
            'wrong_ID_duration_quantiles_frames':dict(zip(['p50','p90','max'],map(float,np.quantile(durations,[.5,.9,1]))))if durations else {'p50':0.,'p90':0.,'max':0.},
            'false_merge_tracks':sum(len(g)>1 for g in gts_by_id.values()),
            'extra_birth_fragments':sum(max(0,len(t)-1)for t in ids_by_gt.values()),
            'identity_index_query':{'precision_micro':tp/retrieved if retrieved else 0.,'recall_micro':tp/den if den else 0.,
                'false_identity_joins':retrieved-tp,'contaminated_retrieved_trajectory_rate':sum(q['false_joins']>0 for q in query)/max(1,sum(q['retrieved']>0 for q in query)),
                'cross_camera_identity_continuity_rate':sum(q['cross_camera_continuity']for q in query if q['expected_cross_camera'])/max(1,sum(q['expected_cross_camera']for q in query)),
                'query_lookup_latency_us_p50_p95':list(map(float,np.quantile(query_times,[.5,.95])/1000))if query_times else [],
                'latency_scope':'prebuilt index lookup only; annotation alignment/evaluation excluded','queries':query}}

def mechanism_correctness(journal,rows):
    mapping,_=IdentityEvaluator.mapping(rows);counts=Counter();events=[]
    with gzip.open(journal,'rt')as f:
        for line in f:
            m=json.loads(line);frame,view=m['key'][1:]
            for row in m['changed_rows']:
                observed=rows.get((frame,view,row));gt=observed['gt']if observed else None
                if gt is None:counts['unknown_GT_changed_rows']+=1;continue
                before_col=m['initial_pairs'].get(str(row),m['initial_pairs'].get(row))
                before_id=m['track_ids'][before_col]if before_col is not None else None
                after_id=m['native_existing_ids'][row]
                old=mapping.get(before_id)==gt;new=mapping.get(after_id)==gt if after_id>=0 else False
                counts['proposal_pair_corrections']+=int(not old and new)
                counts['proposal_pair_regressions']+=int(old and not new)
                events.append({'frame':frame,'view':view,'row':row,'offline_GT':gt,'before_id':before_id,'after_existing_id':after_id,'before_correct':old,'after_correct':new})
    return {'counts':dict(counts),'events':events,'scope':'descriptive initial proposal to post-validation existing identity; birth/recovery excluded; global fixed mapping, not paired causal attribution'}

"""Post-decision TRAIN sidecars. No labels enter a model/solver/scheduler."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

VAL_SCENES=(4,8,12,16,20,24)
LEGAL_VAL=(4,8,12)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def code_fingerprint(repo):
    repo=Path(repo);paths=sorted(set((repo/'worldmot/worldmot').rglob('*.py'))|
        set((repo/'worldmot/analysis').glob('*v4*.py'))|set((repo/'worldmot/scripts').glob('*v4*.py'))|
        set((repo/'worldmot/tests').glob('*.py'))|set((repo/'worldmot/configs').glob('worldobject_v4*')))
    h=hashlib.sha256()
    for path in paths:h.update(str(path.relative_to(repo)).encode()+b'\0'+path.read_bytes())
    return h.hexdigest()


def visual_only(observations):
    """Construct a geometry-free view without inspecting current Ω results."""
    from worldmot.models.world_object_observation import WorldObjectObservation
    return [WorldObjectObservation(o.camera_id,o.detection_id,o.bbox_xywh,o.score,o.roi_feature,o.feature_valid,
        np.full(3,np.nan),np.eye(3)*1e6,None,False,0.,o.timestamp,
        geometry_status='OMEGA_SKIPPED_NO_CURRENT_GEOMETRY',source={'current_geometry_consumed':False}) for o in observations]


def visual_from_detections(detections,timestamp):
    from worldmot.models.world_object_observation import WorldObjectObservation,normalized_feature
    # Match the frozen R2 join -> cache-reader -> observation normalization
    # contract exactly, including its float32 rounding.
    return sorted([WorldObjectObservation(int(camera),o.detection_id,o.bbox_xywh,o.score,
        normalized_feature(normalized_feature(o.detector_roi_feature)),True,np.full(3,np.nan),np.eye(3)*1e6,None,False,0.,int(timestamp),
        geometry_status='OMEGA_NOT_REQUESTED_YET',source={'current_geometry_consumed':False})
        for camera,items in detections.items() for o in items],key=lambda o:o.key)


class HistoryEvidenceObserver:
    def __init__(self,assigner):
        self.calls={};self.original=assigner._history_evidence
        def wrapped(q,o,*args,**kwargs):
            result=self.original(q,o,*args,**kwargs)
            reason=None if result is None else result[1]
            if result is None:
                cfg=assigner.config
                app=None if q.identity_embedding is None or not o.feature_valid else float(q.identity_embedding@o.roi_feature)
                margin=kwargs.get('learned_margin',args[2] if len(args)>2 else None)
                spatial=kwargs.get('spatial_override',args[1] if len(args)>1 else None)
                if app is None:reason='VISUAL_FEATURE_UNAVAILABLE'
                elif app<cfg.min_appearance_similarity and (margin is None or margin<cfg.learned_identity_recovery_logit):
                    reason='APPEARANCE_REJECTION'
                elif (cfg.temporal_geometry and cfg.geometry_policy=='LEGACY_HARD' and q.geometry_valid
                        and o.cross_view_fusion_eligible and spatial is not None and spatial>cfg.geometry_gate):
                    reason='TEMPORAL_GEOMETRY_HARD_REJECTION'
                else:reason='OTHER_OR_CAPACITY_REJECTION'
            self.calls[q.global_id,o.key]=dict(accepted=result is not None,reason=reason)
            return result
        assigner._history_evidence=wrapped


class IdentityAudit:
    def __init__(self):
        self.counts=Counter();self.history_people=defaultdict(set);self.last={}
        self.contingency=Counter();self.gt_gids=defaultdict(set);self.examples=defaultdict(list)
        self.events={};self.step_index=0

    def begin(self,bank):
        self.active={q.global_id for q in bank.active()}
        self.prior={gid:set(self.history_people[gid]) for gid in self.active}
        self.old_spatial={q.global_id:(q.center.copy(),q.covariance.copy()) for q in bank.active()}
        self.old_identity={q.global_id:(None if q.identity_embedding is None else q.identity_embedding.copy()) for q in bank.active()}

    def observe(self,bank,observations,labels,history_calls):
        c=self.counts;self.events={};step=self.step_index;assignments=bank.last_assignments
        labelled=[(o,label,assignments[o.key]) for o,label in zip(observations,labels) if label is not None]
        spatial_writes=set();identity_writes=set();false_births=set()
        for o,label,gid in labelled:
            c['clean_observations']+=1;self.contingency[gid,label]+=1;self.gt_gids[label].add(gid)
            prior=self.prior.get(gid,set())
            is_new=gid not in self.active
            matching=[old for old,people in self.prior.items() if people=={label}]
            if is_new:
                c['new_ID_observations']+=1
                if matching:c['false_birth_despite_clean_active_history']+=1;false_births.add(gid)
            elif prior=={label}:c['correct_historical_ID_accepted']+=1
            else:
                c['wrong_or_mixed_historical_ID_accepted']+=1
                c['wrong_historical_ID_accepted' if len(prior)==1 else
                  'mixed_historical_ID_accepted' if len(prior)>1 else 'unlabelled_historical_ID_accepted']+=1
            if matching:
                rejected=all(not history_calls.get((old,o.key),{}).get('accepted',False) for old in matching)
                c['correct_historical_opportunities']+=1
                c['correct_historical_ID_rejected']+=int(rejected)
                if rejected:
                    reasons={history_calls.get((old,o.key),{}).get('reason','NOT_SEARCHED') for old in matching}
                    for reason in reasons:c['correct_history_rejection_reason/'+reason]+=1
                elif gid not in matching:c['correct_history_admissible_but_not_selected']+=1
            if any(len(people)>1 and label in people for people in [prior]):c['reuse_of_contaminated_ID']+=1
            key=o.camera_id,label
            if key in self.last:
                old,previous=self.last[key];gap=step-previous;change=gid!=old
                c['temporal_links']+=1;c['temporal_ID_changes']+=int(change)
                self.events['temporal',key]=(not change)
                if gap>1:
                    scope='long_gap_over30' if gap>30 else 'gap_2to30'
                    c[scope+'_returns']+=1;c[scope+'_retained_ID']+=int(not change)
            self.last[key]=(gid,step)
            self.events['historical',o.key]=(not is_new and prior=={label})
            q=bank.objects[gid];state=getattr(q,'spatial_state',None)
            if not is_new and prior and prior!={label}:
                wrote=(state is not None and state.last_trusted_geometry_timestamp==o.timestamp
                       and state.uncertainty_status=='TRUSTED_MEASUREMENT')
                if state is None and q.geometry_valid:
                    old_center,old_cov=self.old_spatial[gid]
                    wrote=(not np.array_equal(old_center,q.center,equal_nan=True)
                           or not np.array_equal(old_cov,q.covariance,equal_nan=True))
                if wrote:spatial_writes.add(gid)
                old_embedding=self.old_identity.get(gid)
                if (old_embedding is not None and q.identity_embedding is not None
                        and not np.array_equal(old_embedding,q.identity_embedding)):
                    identity_writes.add(gid)
        for i,(a,la,ga) in enumerate(labelled):
            for b,lb,gb in labelled[i+1:]:
                if a.camera_id==b.camera_id:continue
                same=la==lb;shared=ga==gb
                kind='same_person' if same else 'different_person'
                c[kind+'_pairs']+=1;c['same_person_splits' if same and not shared else
                    'different_person_merges' if not same and shared else 'correct_sync_pairs']+=1
                self.events[kind,(a.key,b.key)]=same==shared
        selected=set(assignments.values())
        for gid in self.active-selected:
            people=self.prior[gid]
            if len(people)==1 and any(label in people for _,label,_ in labelled):c['NULL_despite_valid_observation']+=1
        for o,label,gid in labelled:self.history_people[gid].add(label)
        c['wrong_person_trusted_spatial_writes']+=len(spatial_writes)
        c['wrong_person_long_term_identity_writes']+=len(identity_writes)
        c['false_birth_global_IDs']+=len(false_births)
        c['timestamps']+=1;c['detections']+=len(observations);self.step_index+=1

    def report(self):
        from scipy.optimize import linear_sum_assignment
        gids=sorted({g for g,_ in self.contingency});labels=sorted({l for _,l in self.contingency})
        matrix=np.zeros((len(gids),len(labels)),dtype=np.int64);gi={v:i for i,v in enumerate(gids)};li={v:i for i,v in enumerate(labels)}
        for (g,l),n in self.contingency.items():matrix[gi[g],li[l]]=n
        rows,cols=linear_sum_assignment(-matrix)
        mixed=[g for g,people in self.history_people.items() if len(people)>1]
        minority=sum(int(matrix[gi[g]].sum()-matrix[gi[g]].max()) for g in mixed if g in gi)
        counts=dict(self.counts);counts.update(global_one_to_one_retained_observations=int(matrix[rows,cols].sum()),
            global_IDs_with_clean_labels=len(gids),mixed_global_IDs=len(mixed),minority_observations_in_mixed_IDs=minority,
            GT_people=len(labels),fragmented_GT_people=sum(len(ids)>1 for ids in self.gt_gids.values()),
            total_extra_ID_fragments=sum(max(0,len(ids)-1) for ids in self.gt_gids.values()))
        return {'counts':counts,'scope':'TRAIN clean-label diagnostic, not TrackEval/CVIDF1',
                'gap_semantics':'gap between matched observations, not proven physical occlusion',
                'spatial_pollution_semantics':'wrong-GT-person accepted writes, not absolute 3D position error'}


def contrast_events(reference,variant,counts,on_change=None):
    for key in reference.keys()&variant.keys():
        kind=key[0];a,b=reference[key],variant[key]
        counts[kind+'_events']+=1
        if b and not a:
            counts[kind+'_corrections']+=1
            if on_change:on_change(key,True)
        if a and not b:
            counts[kind+'_regressions']+=1
            if on_change:on_change(key,False)


def timing_summary(rows):
    if not rows:return {}
    return {key:{'P50_s':float(np.percentile([r.get(key,0.) for r in rows],50)),
        'P95_s':float(np.percentile([r.get(key,0.) for r in rows],95)),
        'sum_s':float(sum(r.get(key,0.) for r in rows))} for key in sorted(set().union(*(r.keys() for r in rows)))}

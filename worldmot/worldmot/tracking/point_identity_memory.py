"""Finite-history causal Global IDs using only pretrained point evidence."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass,field
import time
import numpy as np
from .person_point_sampler import PersonPointSampler,PersonPointReference
from .point_to_instance_association import (PointAssociationSettings,point_support,
    exclusive_assignment,bidirectional_matches)


@dataclass
class PointIdentity:
    global_id: int
    last_seen_timestamp: int
    last_camera: int
    latest_detector_bbox_by_camera: dict = field(default_factory=dict)
    reference_points_by_camera: dict = field(default_factory=dict)
    age: int = 0
    missed: int = 0
    status: str = 'NEWBORN'
    source_evidence: list = field(default_factory=list)


class PointIdentityMemory:
    def __init__(self,backend,settings=None,cross_view=True):
        self.backend=backend
        self.settings=settings or PointAssociationSettings()
        self.cross_view=bool(cross_view)
        self.sampler=PersonPointSampler(self.settings.points_per_person)
        self.reset(None)

    def reset(self,scene_id):
        self.scene_id=scene_id
        self.tracks={}
        self.images={}
        self.next_id=1
        self.timestamp=None
        self.last_diagnostics={}

    def _allocate(self):
        gid=self.next_id;self.next_id+=1
        return gid

    def step(self,images,detections,timestamp,scene_id=None):
        start=time.perf_counter();timestamp=int(timestamp)
        if scene_id is not None and scene_id!=self.scene_id:self.reset(scene_id)
        if self.timestamp is not None and timestamp<=self.timestamp:
            raise ValueError('Point memory requires strictly increasing causal timestamps')
        self.timestamp=timestamp
        images={int(v):np.asarray(im) for v,im in images.items()}
        detections={int(v):list(ds) for v,ds in detections.items()}
        if set(images)!=set(detections):raise ValueError('Every camera must have a current detection list, possibly empty')
        for view,ds in detections.items():
            if len({d.detection_id for d in ds})!=len(ds):raise ValueError('Duplicate detection IDs within camera')
            if any(d.view_id!=view for d in ds):raise ValueError('Detection assigned to wrong camera')
        assigned={v:{} for v in detections};decisions={};evidence=[];forwards=[];point_predictions=[]
        old_ids=sorted(self.tracks)
        row_index={gid:i for i,gid in enumerate(old_ids)}
        scores={v:np.zeros((len(old_ids),len(ds)),np.float64) for v,ds in detections.items()}
        best_evidence={}
        grouped=defaultdict(list)
        for gid in old_ids:
            track=self.tracks[gid]
            if track.status=='DELETED':continue
            for view,reference in track.reference_points_by_camera.items():
                key=(view,reference.timestamp)
                if (0<timestamp-reference.timestamp<=self.settings.maximum_reference_gap
                    and key in self.images and len(reference.xy)):
                    grouped[key].append((gid,reference))
        for key,refs in sorted(grouped.items()):
            source_camera,source_time=key
            target_cameras=sorted(images) if self.cross_view else [source_camera]
            keys=[key]+[(v,timestamp) for v in target_cameras]
            rgb=[self.images[key]]+[images[v] for v in target_cameras]
            queries=np.concatenate([r.xy for _,r in refs])
            predicted=self.backend.track(rgb,queries,image_keys=keys,reference_key=key,current_timestamp=timestamp)
            forwards.append({'kind':'TEMPORAL','image_keys':keys,'timings':predicted.timings,
                             'diagnostics':predicted.diagnostics})
            point_predictions.append({'kind':'TEMPORAL','image_keys':keys,'reference_key':key,
                'query_points':queries,'predicted_xy':predicted.xy,'visibility':predicted.visibility,
                'confidence':predicted.confidence,'valid':predicted.valid,
                'source_identities':[(gid,len(ref.xy)) for gid,ref in refs]})
            offset=0
            for gid,ref in refs:
                sl=slice(offset,offset+len(ref.xy));offset+=len(ref.xy)
                for index,view in enumerate(target_cameras,1):
                    for j,det in enumerate(detections[view]):
                        ev=point_support(predicted.xy[index,sl],predicted.visibility[index,sl],
                            predicted.confidence[index,sl],predicted.valid[index,sl],ref.weights,det,
                            self.settings,source_camera,view,timestamp-source_time)
                        ev.update(global_id=gid,reference_timestamp=source_time,kind='TEMPORAL')
                        evidence.append(ev)
                        if ev['association_score']>scores[view][row_index[gid],j]:
                            scores[view][row_index[gid],j]=ev['association_score']
                            best_evidence[view,row_index[gid],j]=ev
        for view in sorted(detections):
            for i,j in exclusive_assignment(scores[view],self.settings.minimum_support):
                gid=old_ids[i];assigned[view][j]=gid
                ev=best_evidence[view,i,j]
                decisions[view,j]={'kind':'TEMPORAL_MATCH' if ev['source_camera']==view else 'TEMPORAL_AND_CROSS_VIEW_MATCH',
                                   'global_id':gid,'evidence':ev}
        # Synchronous newborns are associated before separate IDs are created.
        # Current detections matched in one view may also transfer an existing
        # ID into the other, subject to one-ID-per-camera exclusivity.
        camera_pair=sorted(images)
        if self.cross_view and len(camera_pair)==2 and all(detections[v] for v in camera_pair):
            a,b=camera_pair
            needs_cross=any(len(assigned[v])<len(detections[v]) for v in camera_pair)
            if needs_cross:
                directional={}
                for source,target in [(a,b),(b,a)]:
                    refs=[self.sampler.sample(images[source],d,timestamp) for d in detections[source]]
                    queries=np.concatenate([r.xy for r in refs])
                    keys=[(source,timestamp),(target,timestamp)]
                    predicted=self.backend.track([images[source],images[target]],queries,image_keys=keys,
                        reference_key=keys[0],current_timestamp=timestamp)
                    forwards.append({'kind':'SYNCHRONOUS_CROSS_VIEW','image_keys':keys,
                                     'timings':predicted.timings,'diagnostics':predicted.diagnostics})
                    point_predictions.append({'kind':'SYNCHRONOUS_CROSS_VIEW','image_keys':keys,
                        'reference_key':keys[0],'query_points':queries,'predicted_xy':predicted.xy,
                        'visibility':predicted.visibility,'confidence':predicted.confidence,'valid':predicted.valid,
                        'source_detections':[(d.detection_id,len(ref.xy)) for d,ref in zip(detections[source],refs)]})
                    matrix=np.zeros((len(refs),len(detections[target])),np.float64)
                    offset=0
                    for i,ref in enumerate(refs):
                        sl=slice(offset,offset+len(ref.xy));offset+=len(ref.xy)
                        for j,det in enumerate(detections[target]):
                            ev=point_support(predicted.xy[1,sl],predicted.visibility[1,sl],
                                predicted.confidence[1,sl],predicted.valid[1,sl],ref.weights,det,
                                self.settings,source,target,0)
                            ev.update(source_detection_id=detections[source][i].detection_id,kind='SYNCHRONOUS_CROSS_VIEW')
                            evidence.append(ev);matrix[i,j]=ev['association_score']
                    directional[source,target]=matrix
                for i,j,score in bidirectional_matches(directional[a,b],directional[b,a],self.settings.minimum_support):
                    ga,gb=assigned[a].get(i),assigned[b].get(j)
                    if ga is not None and gb is not None:
                        if ga!=gb:evidence.append({'kind':'CROSS_VIEW_CONFLICT','rejection_reason':'CONFLICTING_EXISTING_IDENTITIES',
                                                   'global_ids':[ga,gb],'detection_ids':[detections[a][i].detection_id,detections[b][j].detection_id]})
                        continue
                    gid=ga if ga is not None else gb
                    if gid is not None:
                        if any(gid in assigned[v].values() and k not in assigned[v] for v,k in [(a,i),(b,j)]):
                            evidence.append({'kind':'CROSS_VIEW_CONFLICT','rejection_reason':'SAME_CAMERA_ID_EXCLUSIVITY','global_id':gid})
                            continue
                    else:gid=self._allocate()
                    assigned[a][i]=gid;assigned[b][j]=gid
                    for view,k in [(a,i),(b,j)]:
                        prior=decisions.get((view,k),{})
                        decisions[view,k]={'global_id':gid,'kind':'TEMPORAL_AND_CROSS_VIEW_MATCH' if prior else 'CROSS_VIEW_MATCH',
                                           'bidirectional_score':score}
        for view,ds in sorted(detections.items()):
            for j,det in enumerate(ds):
                if j not in assigned[view]:
                    gid=self._allocate();assigned[view][j]=gid
                    decisions[view,j]={'global_id':gid,'kind':'NEWBORN',
                        'rejection_reason':'NO_OVERLAP_OR_NO_POINT_SUPPORT' if evidence else 'NO_VALID_HISTORICAL_ANCHOR'}
        seen=set();output={str(v):[] for v in detections}
        for view,ds in sorted(detections.items()):
            assert len(set(assigned[view].values()))==len(ds),'A Global ID cannot occupy two boxes in one camera'
            for j,det in enumerate(ds):
                gid=assigned[view][j];seen.add(gid)
                track=self.tracks.setdefault(gid,PointIdentity(gid,timestamp,view))
                track.last_seen_timestamp=timestamp;track.last_camera=view
                track.latest_detector_bbox_by_camera[view]=tuple(det.bbox_xywh)
                track.reference_points_by_camera[view]=self.sampler.sample(images[view],det,timestamp,gid)
                track.source_evidence=[decisions[view,j]]
                track.status='OBSERVED';track.missed=0
                output[str(view)].append({'global_id':gid,'detection_id':det.detection_id,
                    'bbox_xywh':list(det.bbox_xywh),'score':float(det.score),'visible':True,
                    'evaluation_eligible':True,'output_source':'GMTDET_VGGT_POINT_ASSOC',
                    'association_kind':decisions[view,j]['kind']})
        for gid,track in self.tracks.items():
            track.age+=1
            if gid not in seen:
                track.missed=timestamp-track.last_seen_timestamp
                track.status='DELETED' if track.missed>self.settings.lost_retention else 'LOST'
        self.images={k:im for k,im in self.images.items() if timestamp-k[1]<self.settings.maximum_reference_gap}
        self.images.update({(v,timestamp):im for v,im in images.items()})
        # Lost IDs retain their recorded state but never emit prediction boxes.
        self.last_diagnostics={'point_evidence':evidence,'point_forward_calls':forwards,
            'point_predictions':point_predictions,
            'decisions':[{**d,'view_id':v,'detection_id':detections[v][j].detection_id} for (v,j),d in decisions.items()],
            'raw_candidate_count':sum(map(len,detections.values())),
            'output_count':sum(map(len,output.values())),'lost_ids':[g for g,t in self.tracks.items() if t.status=='LOST'],
            'deleted_ids':[g for g,t in self.tracks.items() if t.status=='DELETED'],
            'memory_s':time.perf_counter()-start}
        assert self.last_diagnostics['raw_candidate_count']==self.last_diagnostics['output_count']
        return {'timestamp':timestamp,'scene_id':self.scene_id,'tracks_2d_by_camera':output,
                'causal':True,**self.last_diagnostics}

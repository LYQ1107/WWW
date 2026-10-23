"""Transparent GMT-ROI/motion controls; no GMT GTA, VGGT or 3D input."""
from dataclasses import dataclass,asdict,field
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Settings:
    appearance: bool=False
    minimum_iou: float=.3
    minimum_cosine: float=.8
    lost_retention: int=30
    maximum_motion_distance: float=2.

    def manifest(self):return asdict(self)


def iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[:2]+a[2:],b[:2]+b[2:])
    intersection=float(np.prod(np.maximum(0,hi-lo)))
    return intersection/max(float(np.prod(a[2:])+np.prod(b[2:])-intersection),1e-12)


def matches(scores):
    if not scores.size:return []
    n,m=scores.shape
    # One independent NULL per row; zero support cannot force an assignment.
    cost=np.concatenate([-scores,np.zeros((n,n))],axis=1)
    r,c=linear_sum_assignment(cost)
    return [(int(i),int(j)) for i,j in zip(r,c) if j<m and scores[i,j]>0]


@dataclass
class Track:
    gid: int
    last_seen: int
    boxes: dict=field(default_factory=dict)
    velocity: dict=field(default_factory=dict)
    features: dict=field(default_factory=dict)
    times: dict=field(default_factory=dict)


class Control:
    def __init__(self,settings):self.settings=settings;self.reset(None)

    def reset(self,scene):
        self.scene=scene;self.timestamp=None;self.tracks={};self.next_id=1

    @staticmethod
    def feature(det):
        if not hasattr(det,'detector_roi_feature'):raise ValueError('A1 requires actual independent detector ROI features')
        x=np.asarray(det.detector_roi_feature,np.float64)
        if x.ndim!=1 or not np.isfinite(x).all() or np.linalg.norm(x)<=0:raise ValueError('Invalid detector ROI')
        return x/np.linalg.norm(x)

    def step(self,detections,timestamp,scene):
        timestamp=int(timestamp)
        if scene!=self.scene:self.reset(scene)
        if self.timestamp is not None and timestamp<=self.timestamp:raise ValueError('Strictly causal timestamps required')
        self.timestamp=timestamp;s=self.settings
        dets={int(v):list(x) for v,x in detections.items()}
        for view,values in dets.items():
            if len({d.detection_id for d in values})!=len(values) or any(d.view_id!=view for d in values):
                raise ValueError('Invalid per-camera detection identity')
        features={(v,j):self.feature(d) for v,values in dets.items() for j,d in enumerate(values)} if s.appearance else {}
        self.tracks={g:t for g,t in self.tracks.items() if timestamp-t.last_seen<=s.lost_retention}
        gids=sorted(self.tracks);assigned={v:{} for v in dets};decisions=[]
        for view,values in sorted(dets.items()):
            scores=np.zeros((len(gids),len(values)))
            for i,gid in enumerate(gids):
                track=self.tracks[gid]
                for j,det in enumerate(values):
                    if view in track.boxes:
                        gap=timestamp-track.times[view]
                        predicted=track.boxes[view]+track.velocity.get(view,np.zeros(4))*gap
                        predicted[2:]=np.maximum(predicted[2:],1.)
                        overlap=iou(predicted,det.bbox_xywh)
                        if not s.appearance:
                            if overlap>=s.minimum_iou:scores[i,j]=overlap
                            continue
                        center_a=predicted[:2]+predicted[2:]/2
                        box=np.asarray(det.bbox_xywh,float);center_b=box[:2]+box[2:]/2
                        motion=float(np.linalg.norm(center_a-center_b)/np.sqrt(np.prod(predicted[2:])))
                        cosine=max(float(features[view,j]@f) for f in track.features.values())
                        if cosine>=s.minimum_cosine and (overlap>=s.minimum_iou or motion<=s.maximum_motion_distance):
                            scores[i,j]=.5*cosine+.5*overlap
                    elif s.appearance:
                        cosine=max(float(features[view,j]@f) for f in track.features.values())
                        if cosine>=s.minimum_cosine:scores[i,j]=.5*cosine
            for i,j in matches(scores):
                assigned[view][j]=gids[i]
                decisions.append({'kind':'APPEARANCE_MOTION_MATCH' if s.appearance else 'IOU_MOTION_MATCH',
                    'view_id':view,'detection_id':values[j].detection_id,'global_id':gids[i],'score':float(scores[i,j])})
        # Current appearance association happens before allocating newborn IDs.
        if s.appearance and len(dets)==2:
            a,b=sorted(dets);scores=np.zeros((len(dets[a]),len(dets[b])))
            for i in range(len(dets[a])):
                for j in range(len(dets[b])):
                    cosine=float(features[a,i]@features[b,j])
                    if cosine>=s.minimum_cosine:scores[i,j]=cosine
            for i,j in matches(scores):
                ga,gb=assigned[a].get(i),assigned[b].get(j)
                if ga is not None and gb is not None:continue
                gid=ga if ga is not None else gb
                if gid is not None and ((ga is None and gid in assigned[a].values()) or (gb is None and gid in assigned[b].values())):continue
                if gid is None:gid=self.next_id;self.next_id+=1
                assigned[a][i]=gid;assigned[b][j]=gid
                decisions.append({'kind':'CURRENT_CROSS_VIEW_ROI_MATCH','global_id':gid,
                    'detections':[[a,dets[a][i].detection_id],[b,dets[b][j].detection_id]],'score':float(scores[i,j])})
        output={str(v):[] for v in dets}
        for view,values in sorted(dets.items()):
            for j,det in enumerate(values):
                if j not in assigned[view]:
                    assigned[view][j]=self.next_id;self.next_id+=1
                    decisions.append({'kind':'NEWBORN','view_id':view,'detection_id':det.detection_id,'global_id':assigned[view][j]})
                gid=assigned[view][j];track=self.tracks.setdefault(gid,Track(gid,timestamp))
                box=np.asarray(det.bbox_xywh,np.float64)
                if view in track.boxes:
                    gap=timestamp-track.times[view];track.velocity[view]=(box-track.boxes[view])/gap
                track.boxes[view]=box;track.times[view]=timestamp;track.last_seen=timestamp
                if s.appearance:track.features[view]=features[view,j]
                output[str(view)].append({'global_id':gid,'detection_id':det.detection_id,
                    'bbox_xywh':list(det.bbox_xywh),'score':float(det.score),'visible':True,
                    'evaluation_eligible':True,'output_source':'GMTDET_2D_APPEARANCE_MOTION' if s.appearance else 'GMTDET_2D_IOU_MOTION'})
            assert len(set(assigned[view].values()))==len(values)
        return {'timestamp':timestamp,'scene_id':scene,'tracks_2d_by_camera':output,'decisions':decisions,
            'raw_candidate_count':sum(map(len,dets.values())),'output_count':sum(map(len,output.values())),
            'lost_ids':[g for g,t in self.tracks.items() if t.last_seen<timestamp],'causal':True,
            'limitations':'No retrospective merging of already separate IDs; no predictions emitted without current detections.'}

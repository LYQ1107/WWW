"""Opt-in precommit policies, using real native candidates and descriptors only."""
from collections import Counter
import time
import torch
from .features import c1_scores


class IdentityRevisionController:
    def __init__(self, cfg):
        self.enabled = bool(cfg.ENABLED)
        self.mode = str(cfg.MODE)
        if self.enabled and self.mode not in ('off', 'keep', 'c1', 'bias2'):
            raise ValueError('CIR mode has not passed native qualification: ' + self.mode)
        if cfg.ENABLE_PROVISIONAL or cfg.ENABLE_REVISION or cfg.ENABLE_RELINK or cfg.REVISION_WINDOW:
            raise ValueError('CIR revision/buffer not qualified in Phase 0')
        self.reset()

    def reset(self):
        self.calls = 0
        self.timestamp = None
        self.video = None
        self.fallbacks = Counter()
        self.score_changed_calls = 0
        self.extra_joint_solves = 0
        self.changed_primary_rows = 0
        self.changed_original_accept_rows = 0
        self.latencies = []
        self.intervention_latencies = []

    def observe(self,instances,records):
        if not self.enabled or self.mode in ('off','keep','bias2'):
            return
        if len(instances)!=len(records) or not records:
            raise ValueError('CIR current image membership mismatch')
        timestamp=int(records[0]['frame_id']);video=int(records[0]['video_id'])
        if self.timestamp is not None and timestamp<=self.timestamp:
            raise ValueError('CIR cannot observe future-to-past groups')
        if self.video is not None and video!=self.video:
            raise ValueError('CIR requires reset between videos')
        self.video=video;self.timestamp=timestamp
        for inst,record in zip(instances,records):
            if int(record['frame_id'])!=timestamp or int(record['video_id'])!=video:
                raise ValueError('Mixed CIR observation groups')
            n=len(inst);device=inst.reid_features.device
            inst.cir_source=torch.stack((
                torch.full((n,),video,device=device,dtype=torch.long),
                torch.full((n,),timestamp,device=device,dtype=torch.long),
                torch.full((n,),int(record['view_id']),device=device,dtype=torch.long),
                torch.full((n,),int(record['image_id']),device=device,dtype=torch.long),
                torch.arange(n,device=device)),dim=1)

    def propose(self, native_scores,instances=None,k=None,candidate_ids=None):
        # KEEP must preserve the very same tensor and consume no RNG.
        self.calls += 1
        if self.mode in ('off','keep') or not native_scores.numel():
            return native_scores
        begin=time.perf_counter()
        try:
            if self.mode=='bias2':
                # Registered fixed-threshold explanation control. It changes
                # acceptance calibration without learning or identity features.
                self.score_changed_calls+=1
                return native_scores+2.
            if instances is None or candidate_ids is None or self.timestamp is None:
                raise ValueError('Missing certified native source metadata')
            query=instances[k];refs=[inst for i,inst in enumerate(instances) if i!=k]
            if not query.has('cir_source') or any(not inst.has('cir_source') for inst in refs):
                raise ValueError('Missing native source provenance')
            qmeta=query.cir_source;meta=torch.cat([inst.cir_source for inst in refs])
            if qmeta.shape!=(len(query),5) or meta.shape[1]!=5:
                raise ValueError('Bad CIR source metadata shape')
            if (qmeta[:,0]!=self.video).any() or (meta[:,0]!=self.video).any() or (qmeta[:,1]!=self.timestamp).any():
                raise ValueError('Foreign video/time evidence')
            history=torch.cat([inst.reid_features for inst in refs])
            ids=torch.cat([inst.track_ids for inst in refs])
            score=c1_scores(native_scores,query.reid_features,history,ids,candidate_ids,meta[:,1],self.timestamp,alpha=2.)
            if not torch.isfinite(score).all():raise ValueError('Nonfinite C1 proposal')
            if torch.equal(score,native_scores):return native_scores
            self.score_changed_calls+=1
            return score
        except ValueError as error:
            # This stage has made no native state or RNG change. An untrusted
            # proposal therefore falls back to the original joint assignment.
            self.fallbacks[str(error)]+=1
            return native_scores
        finally:
            self.latencies.append(time.perf_counter()-begin)

    def note_resolve(self,original_ids,revised_ids):
        self.extra_joint_solves+=1
        self.changed_primary_rows+=int((original_ids!=revised_ids).sum())
        self.changed_original_accept_rows+=int(((original_ids>=0)&(original_ids!=revised_ids)).sum())

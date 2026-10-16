"""Opt-in precommit policies, using real native candidates and descriptors only."""
from collections import Counter
import time
import torch
from .features import c1_scores
from .fragment_ledger import FragmentLedger
from .types import SourceKey,ConflictState
from .conflict_detector import FixedConflictDetector
from .native_evidence import native_conflicts

RISK_MODES=('risk_mlp','risk_grouped')
GUARD_MODES=('guard','margin','threshold','guard_reference',*RISK_MODES)


class IdentityRevisionController:
    def __init__(self, cfg):
        self.enabled = bool(cfg.ENABLED)
        self.mode = str(cfg.MODE)
        if self.enabled and self.mode not in ('off','keep','c1','bias2',*GUARD_MODES):
            raise ValueError('CIR mode has not passed native qualification: ' + self.mode)
        if cfg.ENABLE_PROVISIONAL or cfg.ENABLE_REVISION or cfg.ENABLE_RELINK or cfg.REVISION_WINDOW:
            raise ValueError('CIR revision/buffer not qualified in Phase 0')
        self.commit_snapshot=str(getattr(cfg,'COMMIT_SNAPSHOT','full'))
        if self.commit_snapshot not in ('full','cow_engineering'):
            raise ValueError('Unknown native commit snapshot backend')
        self.audit_only=bool(getattr(cfg,'AUDIT_ONLY',False))
        self.max_history=int(getattr(cfg,'MAX_HISTORY',16))
        self.max_fragments=int(getattr(cfg,'MAX_FRAGMENTS_PER_ID',32))
        self.risk_head=None;self.risk_threshold=None;self.risk_checkpoint_sha256=None
        if self.enabled and self.mode in RISK_MODES:
            from .model import load_native_risk
            path=str(getattr(cfg,'MODEL_PATH',''));digest=str(getattr(cfg,'MODEL_SHA256',''))
            if not path or not digest:raise ValueError('Explicit verified risk checkpoint required; no silent KEEP substitution')
            self.risk_head,self.risk_threshold=load_native_risk(path,digest)
            expected='ordinary_mlp' if self.mode=='risk_mlp' else 'grouped_risk'
            if self.risk_head.architecture!=expected:raise ValueError('Risk mode/checkpoint architecture mismatch')
            self.risk_checkpoint_sha256=digest
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
        self.ledger=FragmentLedger(self.max_history,self.max_fragments) if self.enabled and self.mode in GUARD_MODES else None
        self.detector=FixedConflictDetector()
        self.guard_target_sources=None
        self.veto_edges=0;self.guard_joint_solves=0;self.guard_aborted=0
        self.failed=False;self.failure_events=[]
        self.risk_forward_calls=0;self.risk_scored_query_rows=0
        self.guard_primary_query_rows=0;self.guard_initial_accepted_rows=0
        self.guard_changed_primary_rows=0;self.guard_changed_accepted_rows=0
        self.guard_calls_with_veto=0;self.guard_stable_after_reassociation=0

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
        if self.ledger is not None and not self.failed:self.ledger.observe_group(video,timestamp)
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
        if self.mode in ('off','keep') or self.audit_only or self.failed or not native_scores.numel():
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

    def wants_guard(self):
        return self.enabled and self.mode in GUARD_MODES and not self.audit_only and not self.failed

    def precommit(self,scores,accepted,instances,k,candidates,thresholds,count=True,reference_scores=None):
        if not self.wants_guard() or not scores.numel():return scores
        if self.guard_target_sources is not None and not self.guard_target_sources:return scores
        query=instances[k];refs=[inst for i,inst in enumerate(instances) if i!=k]
        if not query.has('cir_source') or any(not inst.has('cir_source') for inst in refs):
            if self.mode in RISK_MODES or self.mode=='guard_reference':raise ValueError('Certified native reference source required')
            return scores
        raw_reference=self.mode in RISK_MODES or self.mode=='guard_reference'
        if raw_reference and (reference_scores is None or reference_scores.shape!=scores.shape or not torch.isfinite(reference_scores).all() or (reference_scores<=-1e5).any()):
            raise ValueError('Unmasked original C1 reference matrix required for risk evidence')
        if self.mode in RISK_MODES and not (accepted>=1).any():return scores
        if self.mode in ('guard','guard_reference',*RISK_MODES):
            source=torch.cat([inst.cir_source for inst in refs])
            # Normalize within each physical camera so different image sizes
            # do not create a fictitious motion discontinuity.
            def normalized_boxes(inst):
                h,w=inst.image_size
                return inst.pred_boxes.tensor/inst.pred_boxes.tensor.new_tensor([w,h,w,h])
            evidence,features=native_conflicts(query.reid_features,normalized_boxes(query),self.timestamp,
                int(query.cir_source[0,2]) if len(query) else 1,
                torch.cat([inst.reid_features for inst in refs]),torch.cat([normalized_boxes(inst) for inst in refs]),
                torch.cat([inst.track_ids for inst in refs]),source[:,1],source[:,2],candidates,
                reference_scores if raw_reference else scores,accepted)
        else:evidence=None
        risk_logits=None
        if self.mode in RISK_MODES:
            if self.risk_head is None or self.risk_threshold is None:raise RuntimeError('Missing frozen risk head')
            device=query.reid_features.device
            if next(self.risk_head.parameters()).device!=device:self.risk_head.to(device)
            eligible=accepted>=1;risk_logits=features.new_full((len(query),),-float('inf'))
            with torch.inference_mode():risk_logits[eligible]=self.risk_head(features[eligible])
            if not torch.isfinite(risk_logits[eligible]).all():raise RuntimeError('Nonfinite native risk prediction')
            self.risk_forward_calls+=1;self.risk_scored_query_rows+=int(eligible.sum())
        revised=None
        support=torch.stack([(torch.cat([inst.track_ids for inst in refs])==identity).sum() for identity in candidates]).clamp_min(1)
        normalized=scores/support[None,:]
        for row,identity in enumerate(accepted.tolist()):
            if identity<0:continue
            src=query.cir_source[row].tolist();key=(src[0],src[3],src[4])
            if self.guard_target_sources is not None and key not in self.guard_target_sources:continue
            column=int(torch.nonzero(candidates==identity,as_tuple=True)[0][0])
            if self.mode in RISK_MODES:reject=bool(risk_logits[row]>self.risk_threshold)
            elif self.mode in ('guard','guard_reference'):reject=self.detector.classify(evidence[row])[0]==ConflictState.SUSPECT
            elif self.mode=='threshold':reject=bool(scores[row,column]<=1.1*thresholds[column])
            else:
                other=normalized[row].clone();other[column]=-float('inf')
                reject=bool(normalized[row,column]-other.max()<.02) if len(candidates)>1 else False
            if reject:
                if revised is None:revised=scores.clone()
                revised[row,column]=-1e6
                if count:self.veto_edges+=1
        return scores if revised is None else revised

    def committed(self,instance,counts,reids):
        if self.ledger is None or self.failed:return
        for source,identity in zip(instance.cir_source.tolist(),instance.track_ids.tolist()):
            if identity not in counts or identity not in reids or len(reids[identity])!=counts[identity]:
                raise RuntimeError('CIR ledger/native count or Gallery mismatch')
            if reids[identity].cir_source[-1].tolist()!=source:
                raise RuntimeError('CIR native Gallery provenance mismatch')
            self.ledger.register_commit(SourceKey(*source),int(identity))
        self.ledger.validate_invariants()

    def published(self):
        if self.ledger is not None and not self.failed:self.ledger.mark_published(self.timestamp)

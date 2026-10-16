"""Actual original + C1 joint solves with an independent CPU formula check."""
import torch
import torch.nn.functional as F
from verify_cir_off_parity import FullScoreTrace


class C1ScoreTrace(FullScoreTrace):
    def solve(self,cost,*args,**kwargs):
        result=super().solve(cost,*args,**kwargs)
        if self.stack:self.stack[-1]['matrices'].append(cost.detach().cpu().clone())
        return result

    def method(self,name,args,kwargs):
        if kwargs:raise ValueError('Unexpected native call kwargs')
        offset=1 if name=='run_global_tracker_plus' else 0
        instances=args[offset];k=args[offset+3]
        context=self.context(name,instances,k)
        holder=dict(context=context,records=[],matrices=[]);self.stack.append(holder)
        independent=None
        if name=='run_global_tracker_plus' and self.model.cir_controller.mode=='bias2':
            independent=torch.full((len(context['queries']),len({r['track'] for r in context['references']})),2.)
        elif name=='run_global_tracker_plus':
            refs=[inst for i,inst in enumerate(instances) if i!=k]
            features=torch.cat([inst.reid_features.detach().cpu().float() for inst in refs])
            tracks=torch.cat([inst.track_ids.detach().cpu() for inst in refs])
            frames=torch.cat([inst.cir_source[:,1].detach().cpu() for inst in refs])
            query=F.normalize(instances[k].reid_features.detach().cpu().float(),dim=-1)
            timestamp=self.model.cir_controller.timestamp
            candidates=torch.unique(tracks);h=F.normalize(features,dim=-1)
            delta=torch.zeros(len(query),len(candidates))
            for column,identity in enumerate(candidates):
                past=(tracks==identity)&(frames<timestamp)
                if past.any():delta[:,column]=2*(query@F.normalize(h[past].sum(0),dim=0))
            independent=delta
        try:
            result=self.originals[name](*args)
            final=(result if name=='run_memory_tracker' else result[0][k].track_ids).detach().cpu().tolist()
            allowed=(1,2) if name=='run_global_tracker_plus' else (1,)
            if len(holder['records']) not in allowed:raise ValueError('Unexpected actual native solver count')
            last=holder['records'][-1]
            if any(track>=0 and track!=actual for track,actual in zip(last['accepted_track_ids'],final)):
                raise ValueError('Joint native acceptance not reflected in final native commit')
            maximum_error=None
            if independent is not None:
                expected=-holder['matrices'][0]+independent
                actual=-holder['matrices'][-1]
                maximum_error=float((actual-expected).abs().max()) if actual.numel() else 0.
                tolerance=8*torch.finfo(torch.float32).eps*(expected.abs()+3)
                if ((actual-expected).abs()>tolerance).any():raise ValueError('GPU policy differs from independent CPU frozen formula')
            record=dict(**context,solver_sequence=holder['records'],final_track_ids=final,
                c1_independent_CPU_max_absolute_error=maximum_error,
                policy=self.model.cir_controller.mode,
                numeric_check='per-edge abs error <= 8*float32eps*(abs(expected)+3); no score/threshold retuning',
                method_call_index=self.counts.get(name,0),
                scope='actual ordered original and optional C1 native joint solves, original Bank and final commit; no GT')
            self.counts[name]=record['method_call_index']+1
            self.write(self.decisions,record)
            return result
        finally:self.stack.pop()

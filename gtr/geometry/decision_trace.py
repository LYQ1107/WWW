"""Read-only record of actual frozen tracker solver calls, without GT.

Diagnostic-only proposal provenance fields survive Instances slicing/cat.
The solver is called exactly once and its original return objects are retained.
Memory-bank descriptor averages are explicitly a separate stage.
"""
import gzip
import json
import torch


def solver_record(cost,assignment,context,threshold,not_mult_thresh):
    refs=context['references'];ids=sorted({ref['track'] for ref in refs})
    if tuple(cost.shape)!=(len(context['queries']),len(ids)):
        raise ValueError('solver matrix and actual context differ')
    counts={identity:sum(ref['track']==identity for ref in refs) for identity in ids}
    accepted=[-1]*len(context['queries']);pairs=[]
    for row,column in zip(*assignment):
        identity=ids[int(column)]
        # Retain original float32 threshold arithmetic, including strict >.
        boundary=cost.new_tensor(threshold)*(1 if not_mult_thresh else cost.new_tensor(float(counts[identity])))
        legal=bool(-cost[row,column]>boundary)
        if legal:accepted[int(row)]=identity
        pairs.append(dict(row=int(row),track=identity,score=float(-cost[row,column]),threshold=float(boundary),accepted=legal))
    return dict(**context,hungarian_pairs=pairs,accepted_track_ids=accepted)


class DecisionTrace:
    def __init__(self,model,out,tracker_module=None):
        if tracker_module is None:
            from gtr.modeling.meta_arch import gtr_rcnn as tracker_module
        self.model=model;self.module=tracker_module;self.stack=[];self.counts={};self.originals={};self.seen=set()
        self.decisions=gzip.open(out/'actual_decisions.jsonl.gz','xt')
        self.detectors=gzip.open(out/'detector_provenance.jsonl.gz','xt')

    def write(self,stream,record):
        stream.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')

    def inference(self,*args,**kwargs):
        from gtr.geometry.online_runner import clone_instances
        from gtr.modeling.meta_arch.custom_rcnn import CustomRCNN
        from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
        result=self.originals['inference'](*args,**kwargs)
        batch=args[0]
        if len(result)!=len(batch):raise ValueError('per-image inference output mismatch')
        for record,instance in zip(batch,result):
            image=int(record['image_id'])
            if image in self.seen:raise ValueError('repeated image inference in diagnostic')
            self.seen.add(image)
            n=len(instance);device=instance.pred_boxes.tensor.device
            instance.decision_trace_source=torch.stack((torch.full((n,),image,device=device,dtype=torch.long),torch.arange(n,device=device)),dim=1)
            # Exact public postprocessing on a clone; mutable live boxes stay intact.
            public=CustomRCNN._postprocess([clone_instances(instance)],[record],[(0,0)],not_clamp_box=self.model.not_clamp_box)[0]['instances'].to('cpu')
            detections=custom_instances_to_coco_json(public,image)
            for source,detection in zip(public.decision_trace_source.tolist(),detections):
                self.write(self.detectors,dict(source=source,detection=detection))
        return result

    def context(self,stage,instances,k):
        references=[]
        for index,instance in enumerate(instances):
            if index==k:continue
            sources=instance.decision_trace_source.detach().cpu().tolist()
            tracks=instance.track_ids.detach().cpu().tolist()
            references.extend(dict(source=source,track=int(track)) for source,track in zip(sources,tracks))
        return dict(stage=stage,queries=instances[k].decision_trace_source.detach().cpu().tolist(),references=references)

    def method(self,name,args,kwargs):
        if kwargs:raise ValueError('unexpected tracker method keyword call')
        offset=1 if name=='run_global_tracker_plus' else 0
        instances=args[offset];k=args[offset+3]
        context=self.context(name,instances,k);holder=dict(context=context,records=[]);self.stack.append(holder)
        try:
            result=self.originals[name](*args)
            final=(result if name=='run_memory_tracker' else result[0][k].track_ids).detach().cpu().tolist()
            if len(holder['records'])!=1:raise ValueError('expected exactly one actual solver per tracker method')
            record=holder['records'][0]
            if len(final)!=len(record['queries']):raise ValueError('tracker return/query row count mismatch')
            if any(track>=0 and track!=actual for track,actual in zip(record['accepted_track_ids'],final)):
                raise ValueError('accepted solver assignment disagrees with actual tracker return')
            record['final_track_ids']=final
            record['method_call_index']=self.counts.get(name,0);self.counts[name]=record['method_call_index']+1
            record['scope']='actual solver and threshold before optional memory fallback/new-ID allocation; final IDs also recorded; memory averaged-descriptor source tags are provenance, not proof of pure GT identity'
            self.write(self.decisions,record)
            return result
        finally:self.stack.pop()

    def solve(self,cost,*args,**kwargs):
        result=self.originals['linear_sum_assignment'](cost,*args,**kwargs)
        if self.stack:
            current=self.stack[-1];stage=current['context']['stage']
            threshold=self.model.thred_bank if stage=='run_memory_tracker' else self.model.overlap_thresh
            current['records'].append(solver_record(cost,result,current['context'],threshold,self.model.not_mult_thresh))
        return result

    def install(self):
        self.originals['inference']=self.model.inference;self.model.inference=self.inference
        self.originals['linear_sum_assignment']=self.module.linear_sum_assignment;self.module.linear_sum_assignment=self.solve
        for name in ('run_first_tracker_plus','run_global_tracker_plus','run_memory_tracker'):
            self.originals[name]=getattr(self.model,name)
            setattr(self.model,name,lambda *args,_name=name,**kwargs:self.method(_name,args,kwargs))

    def close(self):
        self.decisions.close();self.detectors.close()
        if not self.originals:return
        self.module.linear_sum_assignment=self.originals['linear_sum_assignment']
        for name in ('inference','run_first_tracker_plus','run_global_tracker_plus','run_memory_tracker'):
            setattr(self.model,name,self.originals[name])

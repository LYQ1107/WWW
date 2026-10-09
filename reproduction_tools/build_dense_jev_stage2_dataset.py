"""Dense natural native B1 histories. GT only labels saved inputs after/before decisions."""
import argparse,collections,time
import numpy as np,torch
from scipy.optimize import linear_sum_assignment
from jev_phase13_common import *
from jev_phase13_runtime import *

class OfflineLabels:
    def __init__(self,video,reader):
        d=json.loads(ANNOTATIONS.read_text());self.images={(i['frame_id']-1,i['view_id']-1):i for i in d['images'] if i['video_id']==video};image_ids={i['id'] for i in self.images.values()};self.gt=collections.defaultdict(list)
        for a in d['annotations']:
            if a['image_id'] in image_ids:self.gt[a['image_id']].append(a)
        self.reader=reader;self.video=video;self.aligned={};self.past=collections.defaultdict(set);self.ever=set();self.records=[];self.counts=collections.Counter();self.recall=collections.Counter()
    def current(self,frame,view):
        key=(frame,view)
        if key in self.aligned:return self.aligned[key]
        image=self.images[key];p=self.reader.load(self.video,frame,view);boxes=p['pred_boxes'].cpu().numpy().copy();boxes[:,[0,2]]*=image['width']/p['image_size'][1];boxes[:,[1,3]]*=image['height']/p['image_size'][0];gt=self.gt[image['id']];result=[None]*len(boxes)
        if len(boxes) and gt:
            g=np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in gt]);lo=np.maximum(boxes[:,None,:2],g[None,:,:2]);hi=np.minimum(boxes[:,None,2:],g[None,:,2:]);inter=np.maximum(hi-lo,0).prod(-1);area=np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(-1);ga=np.maximum(g[:,2:]-g[:,:2],0).prod(-1);iou=inter/np.maximum(area[:,None]+ga[None]-inter,1e-9)
            rows,cols=linear_sum_assignment(-iou)
            for row,col in zip(rows,cols):
                if iou[row,col]>=.5:result[row]=int(gt[col]['instance_id'])
        self.aligned[key]=result;return result
    def before(self,**kw):
        x,refs,c,task=kw['batch'],kw['refs'],kw['context'],kw['task'];targets=self.current(c['frame'],c['view']);rows=c.get('rows',list(range(len(targets))));q=len(rows);k=len(refs);positive=torch.zeros(q,k+1,dtype=torch.bool);known=torch.zeros_like(positive);gold=torch.full((q,),k,dtype=torch.long);labelled=torch.zeros(q,dtype=torch.bool)
        if c.get('first') and task==0 and not self.past:
            bootstrap=self.current(0,1-c['view'])
            for ref in refs:
                target=bootstrap[ref-1]
                if target is not None:self.past[ref].add(target);self.ever.add(target)
        for r,row in enumerate(rows):
            target=targets[row]
            if target is None:self.counts['unknown_current_GT']+=1;continue
            for col,ref in enumerate(refs):
                votes=self.past[ref]
                if len(votes)==1:
                    known[r,col]=True;positive[r,col]=target in votes
                elif len(votes)>1:self.counts['ambiguous_history_options']+=1
            if positive[r,:k].any():
                gold[r]=int(torch.where(positive[r])[0][0]);labelled[r]=True;known[r,-1]=True;self.counts['normal_active_positive' if task==0 else 'real_stale_positive']+=1
            elif task==0:
                gold[r]=k;labelled[r]=True;positive[r,k]=True;known[r,k]=True;self.counts['MATCH_DEFER_labels']+=1
            elif target not in self.ever:
                gold[r]=k;labelled[r]=True;positive[r,k]=True;known[r,k]=True;self.counts['genuine_NEW_labels']+=1
            else:self.counts['REACT_correct_history_missing_or_ambiguous']+=1
            clean_support=any(target in self.past[t] and len(self.past[t])==1 for t in refs);self.recall['active_known_rows' if task==0 else 'stale_known_rows']+=1;self.recall['active_clean_support' if task==0 else 'stale_clean_support']+=int(clean_support)
        self.counts['MATCH_rows' if task==0 else 'REACT_rows']+=q;self.counts['supervised_rows']+=int(labelled.sum());self.counts['candidate_options']+=q*k
        if q:self.records.append({'key':[self.video,c['frame'],c['view']],'task':task,'refs':refs,'rows':rows,'inputs':{k:v.detach().cpu().clone() for k,v in x.items()},'positive':positive,'known_options':known,'targets':gold,'supervised':labelled})
    def after(self,**kw):
        key=(kw['frame'],kw['view']);labels=self.current(*key);inst=kw['instances'][-1]
        if kw['first'] and not self.past:
            # Native largest-camera bootstrap has no association callback.
            other=1-kw['view'];initial=kw['instances'][0];bootstrap=self.current(kw['frame'],other)
            for ref,target in zip(initial.track_ids.tolist(),bootstrap):
                if target is not None:self.past[ref].add(target);self.ever.add(target)
        for ref,target in zip(inst.track_ids.tolist(),labels):
            if target is not None:self.past[ref].add(target);self.ever.add(target)

def main(video):
    allowed(video);protect();torch.set_num_threads(1);torch.manual_seed(20261009);source=binding();assert not source['dirty']
    values,frames,reader=cache_inputs(video);out=OUT/'dense_native_v1'/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True);assert not (out/'RESULT.json').exists()
    labels=OfflineLabels(video,reader);model=build_tracker(video);model.jev_stage2_executor.observer=labels.before;model.jev_stage2_executor.commit_observer=labels.after;t=time.monotonic()
    with torch.no_grad():raw,id_count=run(model,values,frames)
    path=out/'DATASET.pth';torch.save({'records':labels.records,'scope':'all actual B1-native decisions with fresh true Stage1 features; GT offline labels, histories neither teacher-forced nor GTA generated','video':video},path)
    save(out/'RESULT.json',{'status':'COMPLETE','binding':source,'video':video,'frames':frames,'record_groups':len(labels.records),'counts':dict(labels.counts),'recall':dict(labels.recall),'DATASET':{'path':str(path),'SHA256':sha(path),'bytes':path.stat().st_size},'natural_native_ids':id_count,'known_GT_targets_seen':len(labels.ever),'contaminated_historical_ids':sum(len(s)>1 for s in labels.past.values()),'GTA_throw_mock':True,'elapsed_seconds':time.monotonic()-t,'heldout':'SEALED','Full24':False,'official_TEST':False});print('DENSE_NATIVE_VIDEO_COMPLETE',video,len(labels.records),dict(labels.counts),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);main(p.parse_args().video)

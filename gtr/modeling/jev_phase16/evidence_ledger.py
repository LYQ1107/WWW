"""Append-only provenance for real native commits, separate from matching views."""
import copy
import torch
import torch.nn.functional as F

class EvidenceLedger:
    def __init__(self):
        self.records={};self._cache={};self.last_payload=None

    def append(self,identity,feature,box,image_size,frame,view,row,score):
        identity=int(identity);key=(int(frame),int(view),int(row))
        records=self.records.setdefault(identity,[])
        assert not any(r['key']==key for r in records[-2:]),'duplicate committed observation'
        records.append(dict(key=key,feature=feature[:1024].detach().clone(),
            box=box.detach().clone()/box.new_tensor([image_size[1],image_size[0],image_size[1],image_size[0]]),
            score=float(score)))
        self._cache.pop(identity,None);self.last_payload=(int(frame),int(view))

    def append_instance(self,inst,frame,view):
        assert inst.has('track_ids') or len(inst)==0
        for row,identity in enumerate(inst.track_ids.tolist() if inst.has('track_ids') else []):
            self.append(identity,inst.reid_features[row],inst.pred_boxes.tensor[row],
                inst.image_size,frame,view,row,inst.scores[row])

    @classmethod
    def from_prefix(cls,prefix):
        result=cls();boundary=tuple(prefix['key'][1:]);instances=prefix['instances']
        # The actual frame-zero native bootstrap commits the largest camera
        # first, including camera1 on ties. Gallery order must match this.
        if len(instances)>=2:
            bootstrap=max((len(instances[v]),v) for v in [0,1])[1]
            first_order=[bootstrap,1-bootstrap]
        else:first_order=list(range(len(instances)))
        order=first_order+list(range(2,len(instances)))
        for index in order:
            frame,view=divmod(index,2);inst=instances[index]
            if (frame,view)>=boundary and not (frame==0 and inst.has('track_ids')):continue
            if not inst.has('track_ids'):assert len(inst)==0;continue
            result.append_instance(inst,frame,view)
        return result

    def matrix(self,identity):
        identity=int(identity)
        if identity not in self._cache:
            value=torch.stack([r['feature'] for r in self.records[identity]])
            self._cache[identity]=(value,F.normalize(value,dim=-1))
        return self._cache[identity]

    def verify_raw_galleries(self,galleries):
        assert set(self.records)==set(galleries),'ledger and actual Gallery identities differ'
        for identity,gallery in galleries.items():
            raw,_=self.matrix(identity)
            assert torch.equal(raw,gallery.reid_features[:,:1024]),('raw Gallery provenance mismatch',identity,len(raw),len(gallery))

    def state_dict(self):
        return copy.deepcopy(dict(version=1,records=self.records,last_payload=self.last_payload))

    def load_state_dict(self,state):
        assert state['version']==1
        self.records=copy.deepcopy(state['records']);self.last_payload=state['last_payload'];self._cache={}

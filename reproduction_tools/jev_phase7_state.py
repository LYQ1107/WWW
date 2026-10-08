"""Digest every mutable state field; immutable perception content is cached."""
import hashlib,json
import torch

class CompleteStateDigest:
    def __init__(self):self.perception_hashes={}
    def _walk(self,digest,value):
        if torch.is_tensor(value):
            tensor=value.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype),list(tensor.shape))).encode());digest.update(tensor.numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=lambda k:(type(k).__name__,str(k))):
                digest.update(str((type(key).__name__,key)).encode())
                if key=='perception':
                    payload=value[key];identity=tuple(int(payload[k])for k in ('video_id','frame','view'))
                    if identity not in self.perception_hashes:
                        part=hashlib.sha256();self._walk(part,payload);self.perception_hashes[identity]=part.digest()
                    digest.update(self.perception_hashes[identity])
                else:self._walk(digest,value[key])
        elif isinstance(value,(list,tuple)):
            digest.update(type(value).__name__.encode())
            for item in value:self._walk(digest,item)
        elif isinstance(value,set):self._walk(digest,sorted(value))
        elif value is None or isinstance(value,(bool,str,int,float)):
            digest.update(json.dumps(value,allow_nan=False).encode()+b';')
        else:raise TypeError(type(value))
    def __call__(self,state,metadata):
        digest=hashlib.sha256();self._walk(digest,vars(state));self._walk(digest,metadata);return digest.hexdigest()

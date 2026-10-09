"""Lossless bounded dense artifacts: preserve tensor bits without redundant disk copies."""
import io
import lzma
import shutil
from pathlib import Path
import torch

FILTERS=[{'id':lzma.FILTER_LZMA2,'preset':3,'dict_size':32*1024*1024}]

def save_dense(path,value,reserve=3000000000):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp');assert not path.exists()
    with lzma.open(temporary,'wb',filters=FILTERS) as stream:torch.save(value,stream)
    assert shutil.disk_usage(path.parent).free>=reserve,'disk reserve below frozen3GB after dense serialization'
    temporary.replace(path)

def load_dense(path):
    path=Path(path)
    if path.suffix=='.xz':
        with lzma.open(path,'rb') as stream:data=stream.read()
        return torch.load(io.BytesIO(data),map_location='cpu')
    return torch.load(path,map_location='cpu')

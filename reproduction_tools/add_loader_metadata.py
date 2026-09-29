"""Add unused legacy loader metadata omitted by official creat_json.py."""
from pathlib import Path
from collections import Counter
import hashlib, json, shutil
r=Path(__file__).resolve().parents[1]
backup=r/'backup/official_converter_outputs'
backup.mkdir(exist_ok=True)
records=[]
for name in ['train_stage1','train','test']:
    p=r/'datasets/VisionTrack/annotations'/f'{name}.json'
    original=backup/p.name
    if original.exists():raise RuntimeError('Refusing to overwrite converter snapshot')
    shutil.copy2(p,original)
    data=json.loads(p.read_text())
    lengths=Counter((im['video_id'],im['view_id']) for im in data['images'])
    for im in data['images']:
        n=lengths[im['video_id'],im['view_id']]
        if 'ids_range' in im or 'len_seq' in im:raise RuntimeError('Metadata already provided; inspect before adaptation')
        im['ids_range']=n;im['len_seq']=n
    p.write_text(json.dumps(data))
    records.append({'file':str(p),'official_converter_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'adapted_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'image_records':len(data['images']),'fields_added':['ids_range','len_seq'],'values':'number of frames in the image video/view','annotations_unchanged':True})
(r/'manifests/loader_metadata_adaptation.json').write_text(json.dumps(records,indent=2))

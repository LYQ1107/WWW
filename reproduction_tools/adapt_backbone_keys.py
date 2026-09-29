"""Adapt only two pretrained parameter names to the repository's DFConv2d wrapper."""
from pathlib import Path
import torch,hashlib,json
r=Path(__file__).resolve().parents[1];src=r/'checkpoints/backbone/CH_FPN_1x.pth';dst=r/'checkpoints/backbone/CH_FPN_1x_key_adapted.pth'
assert not dst.exists()
x=torch.load(src,map_location='cpu');state=x['model'];mapping={f'proposal_generator.centernet_head.bbox_tower.9.{s}':f'proposal_generator.centernet_head.bbox_tower.9.conv.{s}' for s in ['weight','bias']}
for a,b in mapping.items():assert a in state and b not in state;state[b]=state.pop(a)
torch.save(x,dst)
y=torch.load(dst,map_location='cpu');original=torch.load(src,map_location='cpu')
assert set(y)==set(original)
assert len(y['model'])==len(original['model'])
for k,v in original['model'].items():assert torch.equal(v,y['model'][mapping.get(k,k)])
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={'source':str(src),'source_sha256':h(src),'derived':str(dst),'derived_sha256':h(dst),'mapping':mapping,'tensor_values':'Every model tensor bitwise unchanged; names only','reason':'Official checkpoint uses plain-convolution names; configured DFConv2d nests these same-shaped parameters under conv. Default loader drops both tensors, causing near-empty proposals and NaN on tested training clip. Offset initialization remains official.'}
(r/'manifests/backbone_key_adaptation.json').write_text(json.dumps(record,indent=2))
with (r/'manifests/checkpoints.tsv').open('a') as f:f.write(f'backbone_key_adapted\t{dst}\t{dst.stat().st_size}\t{h(dst)}\t\n')
print(json.dumps(record,indent=2))

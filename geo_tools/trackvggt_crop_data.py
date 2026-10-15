"""Bounded native-box magnification, never a new physical camera or GT ROI."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from common import *


def crop_rectangle(box,width,height,scale=1.5):
    x1,y1,x2,y2=map(float,box)
    if not np.isfinite([x1,y1,x2,y2]).all() or x2<=x1 or y2<=y1:raise ValueError('invalid native box')
    cx,cy=(x1+x2)/2,(y1+y2)/2;w,h=(x2-x1)*scale,(y2-y1)*scale
    rect=[max(0,math.floor(cx-w/2)),max(0,math.floor(cy-h/2)),min(width,math.ceil(cx+w/2)),min(height,math.ceil(cy+h/2))]
    if rect[2]<=rect[0] or rect[3]<=rect[1]:return None
    return rect


def crop_image(image,rect,resolution=224):
    from gtr.geometry.roi_feature_extractor import PadMapping
    if resolution not in (224,336):raise ValueError('bounded patch14 crop resolution only')
    local=image.crop(tuple(rect)).convert('RGB');mapping=PadMapping(local.width,local.height,target=resolution)
    width,height=mapping.resized
    if min(width,height)<14:raise ValueError('degenerate crop aspect ratio')
    local=local.resize((width,height),Image.Resampling.BICUBIC)
    pixels=torch.from_numpy(np.asarray(local).copy()).permute(2,0,1).float()/255
    dh,dw=resolution-height,resolution-width
    pixels=F.pad(pixels,(dw//2,dw-dw//2,dh//2,dh-dh//2),value=1.)
    return pixels[None,None],mapping


def crop_target_pool(layers,attention,query,rect,resolution=224):
    from gtr.geometry.roi_feature_extractor import PadMapping,patch_map,pool_roi
    from gtr.geometry.identity_guided_sampler import patch_regions,greedy_2d,normalize_attention
    from gtr.geometry.trackvggt_features import LAYERS
    mapping=PadMapping(rect[2]-rect[0],rect[3]-rect[1],target=resolution);device=layers[4].device
    offset=torch.tensor(rect[:2],device=device,dtype=torch.float32)
    local_box=torch.tensor(query['box'],device=device,dtype=torch.float32)-offset.repeat(2)
    xy,inside,context,valid=patch_regions(mapping,local_box,device=device)
    saliency=attention[0,0].float();normalized=torch.zeros_like(saliency);region=inside|context
    normalized[region]=normalize_attention(saliency[region])
    spatial=greedy_2d(inside.float(),xy,inside,torch.zeros_like(context),24,context_cap=0)
    guided=greedy_2d(normalized+inside.float(),xy,inside,context,8,context_cap=8)
    pool=torch.cat((spatial,guided)).unique(sorted=True)
    if len(pool)>32 or not valid[pool].all():raise ValueError('crop compact pool bounds')
    roi=[]
    for layer in LAYERS:
        feature_map=patch_map(layers[layer][0,0],0,(resolution,resolution))
        value,_=pool_roi(feature_map,mapping.boxes(local_box[None]),'roi_align');roi.append(value[0].cpu().numpy().astype(np.float16))
    arrays={f'layer_{layer}':layers[layer][0,0,pool].float().cpu().numpy().astype(np.float16) for layer in LAYERS}
    arrays.update(xy=(xy[pool]+offset).cpu().numpy(),inside=inside[pool].cpu().numpy(),attention=saliency[pool].cpu().numpy(),
        patch_indices=pool.cpu().numpy(),roi=np.stack(roi))
    count=int(inside.sum());bucket='0' if count==0 else '1-3' if count<4 else '4-15' if count<16 else '16+'
    return arrays,dict(raw_patch_centers=count,patch_bin=bucket,valid_pool_tokens=len(pool),all_selected_inside_content=True)


def merge_target_arrays(full,local,variant):
    if variant=='full':return full
    if variant=='local':return local
    if variant!='combined':raise ValueError('unknown target feature variant')
    # Same current observation, different raster scales. Positions remain original
    # image coordinates; no fictitious physical camera or duplicate identity node.
    result={name:np.concatenate([full[name],local[name]],axis=0) for name in full if name!='roi'}
    def normalized(values):
        return (values-values.min())/max(float(values.max()-values.min()),1e-8) if len(values) else values
    result['attention']=np.concatenate([normalized(full['attention']),normalized(local['attention'])])
    result['roi']=(full['roi'].astype(np.float32)+local['roi'].astype(np.float32))/2
    return result


class CropEventData:
    """Overlay on unchanged V3 inputs, using one fixed256-fit/300-val comparison."""
    def __init__(self,base,plan_path,variant):
        self.base=base;self.variant=variant;self.plan_path=Path(plan_path);self.plan=json.loads(self.plan_path.read_text())
        if self.plan['status']!='LOCKED_OBSERVATION_ONLY_CROP_COMPARISON':raise ValueError('unlocked crop plan')
        if sha256(base.cohort_path)!=self.plan['base_cohort_sha256']:raise ValueError('crop base cohort differs')
        fit={tuple(key) for key in self.plan['fit_keys']};val={tuple(key) for key in self.plan['validation_keys']}
        self.fit=[i for i in base.fit if tuple(base.items[i]['key']) in fit];self.validation=list(base.validation)
        if len(self.fit)!=len(fit) or {tuple(base.items[i]['key']) for i in self.validation}!=val:raise ValueError('crop fixed query scope')
        self.source_artifacts=dict(base.source_artifacts);self.source_artifacts[str(self.plan_path)]=sha256(self.plan_path)
        path=Path(self.plan['feature_manifests'][base.backbone]);manifest=json.loads(path.read_text())
        if manifest['status']!='COMPLETE_BOUNDED_REAL_CROP_POOL' or manifest['plan_sha256']!=sha256(self.plan_path):raise ValueError('crop features incomplete')
        self.source_artifacts[str(path)]=sha256(path);self.crop_rows={tuple(row['key']):row for row in manifest['rows']}
        if set(self.crop_rows)!=fit|val:raise ValueError('crop pool query scope differs')
        for row in self.crop_rows.values():
            if sha256(row['file'])!=row['sha256']:raise ValueError('crop token changed')
            self.source_artifacts[row['file']]=row['sha256']

    def __getattr__(self,name):return getattr(self.base,name)

    def effective_patch_count(self,index):
        key=tuple(self.items[index]['key']);full=self.base.rows[key]['raw_patch_centers'];local=self.crop_rows[key]['raw_patch_centers']
        return full if self.variant=='full' else local if self.variant=='local' else full+local

    def get(self,index,device='cpu'):
        features,target=self.base.get(index,device);key=tuple(self.items[index]['key'])
        if self.variant=='full':return features,target
        with np.load(self.crop_rows[key]['file']) as values:local={name:values[name].copy() for name in values.files}
        array=merge_target_arrays(self.base.arrays(key),local,self.variant)
        for field in ('attention','xy','inside','roi'):
            value=torch.from_numpy(array[field].copy());features[field]=(value.bool() if field=='inside' else value.float()).to(device)
        features['layers']=[torch.from_numpy(array['layer_'+str(layer)].copy()).float().to(device) for layer in (4,11,17,23)]
        # Keep original full-image size bins for a comparable reporting denominator.
        return features,target

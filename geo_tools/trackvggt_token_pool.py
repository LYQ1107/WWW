"""Shared exact V2 full-image target pool; no history/GT enters sampling."""
import numpy as np
import torch
from common import *


def full_image_target_pool(layers,attention,view,query):
    from gtr.geometry.roi_feature_extractor import PadMapping,patch_map,pool_roi
    from gtr.geometry.identity_guided_sampler import patch_regions,greedy_2d,normalize_attention
    from gtr.geometry.trackvggt_features import LAYERS
    mapping=PadMapping(query['width'],query['height']);device=layers[4].device
    xy,inside,context,valid=patch_regions(mapping,query['box'],device=device)
    saliency=attention[0,view].float();normalized=torch.zeros_like(saliency);region=inside|context
    normalized[region]=normalize_attention(saliency[region])
    guided=greedy_2d(normalized+inside.float(),xy,inside,context,64,context_cap=32)
    roi_only=greedy_2d(inside.float(),xy,inside,torch.zeros_like(context),64,context_cap=0)
    pool=torch.cat((guided,roi_only)).unique(sorted=True)
    original_box=torch.tensor([query['box']],device=device);roi=[]
    for layer in LAYERS:
        fm=patch_map(layers[layer][0,view],0,(518,518));value,_=pool_roi(fm,mapping.boxes(original_box),'roi_align')
        roi.append(value[0].cpu().numpy().astype(np.float16))
    arrays={f'layer_{layer}':layers[layer][0,view,pool].float().cpu().numpy().astype(np.float16) for layer in LAYERS}
    arrays.update(xy=xy[pool].cpu().numpy(),inside=inside[pool].cpu().numpy(),attention=saliency[pool].cpu().numpy(),patch_indices=pool.cpu().numpy(),roi=np.stack(roi))
    count=int(inside.sum());bucket='0' if count==0 else '1-3' if count<4 else '4-15' if count<16 else '16+'
    return arrays,dict(raw_patch_centers=count,patch_bin=bucket,valid_pool_tokens=len(pool),all_selected_inside_content=bool(valid[pool].all()))

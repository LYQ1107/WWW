"""Exact historical model bridge, empty payloads and cache invalidation contracts."""
import copy
import importlib.util
import torch
from jev_phase14_common import *
from gtr.modeling.jev_phase14.model import ReliableIdentityPolicy
from gtr.modeling.jev_phase14.memory import CachedIdentityMemory
from gtr.modeling.jev_stage2.memory import IdentityMemory
from detectron2.structures import Instances,Boxes

def main():
    protect();torch.set_num_threads(1);torch.manual_seed(20261009)
    source=OUT/'source_pilot_v2/gtr/modeling/jev_phase14/model.py'
    spec=importlib.util.spec_from_file_location('phase14_original_training_model',source)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    item=read(REPORTS/'QUESTION_SUPERVISION_ELIGIBILITY.json')['videos'][1]
    record=torch.load(item['source_DATASET']['path'],map_location='cpu')['records'][0]
    x=record['inputs'];models={}
    for variant in ['full','fixed_question','multi_question','set_transformer','motip','shared_mlp']:
        a=original.ReliableIdentityPolicy(variant).eval();b=ReliableIdentityPolicy(variant).eval();b.load_state_dict(a.state_dict(),strict=True)
        with torch.no_grad():
            old=a.details(x);new=b.details(x)
            assert all(torch.equal(old[k],new[k]) for k in old),(variant,'training bridge')
            empty={k:v.clone() for k,v in x.items()}
            for k in ['detection_visual','detection_meta','question_mask','question_type']:empty[k]=empty[k][:,:0]
            for k in ['pair_evidence','legal']:empty[k]=empty[k][:,:0]
            assert b(empty).shape==(1,0,x['identity_mask'].shape[1]+1)
        models[variant]=dict(nonempty_bitwise_equal=True,empty_payload=True,parameter_keys_unchanged=True)
    current=Instances((1080,1920));current.pred_boxes=Boxes(torch.tensor([[10.,20.,40.,80.]]));current.scores=torch.tensor([.8]);current.reid_features=torch.randn(1,1024)
    gallery=Instances((1080,1920));gallery.reid_features=torch.randn(2,1024)
    galleries={1:gallery};reference=IdentityMemory();cached=CachedIdentityMemory()
    def equal():
        x=reference.build(current,[1],galleries,20,0,{1});y=cached.build(current,[1],galleries,20,0,{1})
        assert all(torch.equal(x[k],y[k]) for k in x),[k for k in x if not torch.equal(x[k],y[k])]
    for memory in [reference,cached]:
        for view in [0,1]:memory.update(1,gallery.reid_features[view],current.pred_boxes.tensor[0],current.image_size,view,view)
    equal();equal();assert cached.cache_hits>=1
    before=cached.cache_misses;gallery.reid_features.add_(.03);equal();assert cached.cache_misses==before+1
    for memory in [reference,cached]:memory.meta[1]['views'][0]['sum'].add_(.1)
    equal();before=cached.cache_misses
    gallery.reid_features=gallery.reid_features.clone();equal();assert cached.cache_misses==before+1
    for memory in [reference,cached]:memory.update(1,current.reid_features[0],current.pred_boxes.tensor[0],current.image_size,12,0)
    assert not cached._history_cache;equal()
    cached.load_state_dict(reference.state_dict());assert not cached._history_cache;equal()
    for memory in [reference,cached]:memory.meta.pop(1)
    for memory in [reference,cached]:memory.update(1,current.reid_features[0],current.pred_boxes.tensor[0],current.image_size,19,1)
    equal()
    save(REPORTS/'CACHE_INVALIDATION_TESTS.json',dict(status='PASS',binding=binding(),
        training_source_model_SHA256=sha(source),nonempty_parameter_compatible_bridge=models,
        gallery_inplace_change=True,camera_sum_inplace_change=True,tensor_replacement_identity_guard=True,
        committed_append_invalidates=True,resume_invalidates=True,reference_reuse_invalidates=True,
        exact_causal_tensor_difference=0.,scope='targeted numeric and semantic contracts; full native loop proof is separate'))
    print('PHASE14_CACHE_INVALIDATION_AND_MODEL_BRIDGE_PASS',flush=True)

if __name__=='__main__':main()

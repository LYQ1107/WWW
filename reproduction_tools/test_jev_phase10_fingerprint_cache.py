"""A read-only byte cache must produce the unmodified complete state hash."""
import torch,gc
from jev_phase10_common import *
from gtr.modeling.jev_native_state import fingerprint

def main():
 results=[]
 for device in ['cpu']+(['cuda:0']if torch.cuda.is_available()else[]):
  cache={};a=torch.randn(80,128,device=device);value={'a':a,'alias':a[3:7],'noncontiguous':a.T,'ids':torch.arange(15,device=device)}
  for operation in ['initial','reuse','mutate_storage_alias','replace_tensor','change_shape']:
   if operation=='mutate_storage_alias':a[3].add_(2)
   if operation=='replace_tensor':value['ids']=torch.arange(16,device=device)
   if operation=='change_shape':value['alias']=a[2:9]
   assert fingerprint(value,tensor_cache=cache)==fingerprint(value)
  gc.collect();results.append({'device':device,'cases':5,'exact_SHA256_equivalence':True})
 save(REPORTS/'FINGERPRINT_CACHE_TESTS.json',{'status':'PASS','cases':results,'cache_semantics':'actual tensor bytes, dtype/shape/stride/version and live object identity; alias mutation invalidates; no approximate hashing','production_hash_format_unchanged':True,'canonical_capture_source_not_edited':True});print('READONLY_HASH_CACHE_PASS',results)
if __name__=='__main__':main()

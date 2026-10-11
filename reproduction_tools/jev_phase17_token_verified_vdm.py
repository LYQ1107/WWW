"""Local adapter: cache only a verified common prefix of true full tokenizations."""
import torch
from vdm.models.vdm_model import VDM,PreparedGroup
from vdm.models.prompts import prefix_text,suffix_text
class TokenVerifiedVDM(VDM):
 def prepare_group(self,image,questions,shared_context=''):
  pre=prefix_text(shared_context);base=self.processor(text=[pre],images=[image],return_tensors='pt');full=[];positions=[];counts=[]
  for q in questions:
   options=list(q['candidates']);assert 2<=len(options)<=16
   enc=self.processor(text=[pre+suffix_text(q['instruction'],options,q['qtype'])],images=[image],return_tensors='pt')
   assert torch.equal(enc['pixel_values'],base['pixel_values']) and torch.equal(enc['image_grid_thw'],base['image_grid_thw'])
   ids=enc['input_ids'][0];full.append(ids);counts.append(len(options));mm=(ids[None]==self.image_token_id).int()
   pos,_=self.vl_model.get_rope_index(ids[None].to(self.device_str),mm_token_type_ids=mm.to(self.device_str),image_grid_thw=base['image_grid_thw'].to(self.device_str));positions.append(pos[:,0])
  # Cap at the public-context token length. Never cache another question's text.
  length=min(len(base['input_ids'][0]),min(map(len,full)))
  while length>0 and not all(torch.equal(full[0][:length],ids[:length]) for ids in full):length-=1
  # A single question also must share only the genuine public-context prefix.
  while length>0 and not torch.equal(full[0][:length],base['input_ids'][0][:length]):length-=1
  assert length>0 and bool((full[0][:length]==self.image_token_id).any())
  prefix=full[0][:length];suffixes=[ids[length:] for ids in full]
  assert all(torch.equal(torch.cat([prefix,suffix]),ids) for suffix,ids in zip(suffixes,full))
  g=PreparedGroup(prefix,suffixes,base['pixel_values'],base['image_grid_thw'],positions,[q['qtype'] for q in questions],counts,self.image_token_id)
  g.token_boundaries_exact=True;g.public_prefix_tokens_removed=len(base['input_ids'][0])-length
  return g

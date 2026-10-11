"""Actual processor token boundaries and explicit cross-state cache identity guard."""
from jev_phase17_common import *
from PIL import Image
from transformers import AutoProcessor
import torch
sys.path.insert(0,str(OUT/'external/Visual-Jev/code'))
from vdm.models.prompts import prefix_text,suffix_text
if __name__=='__main__':
 protect();base=read(OUT/'teacher_models/MANIFEST.json')[0];processor=AutoProcessor.from_pretrained(base['folder']);processor.image_processor.max_pixels=65536;processor.image_processor.min_pixels=3136
 seen=set();tests=[]
 for pair in read(OUT/'teacher_data/PAIRS.json'):
  wire=pair['inference'];group=wire['group']
  if group in seen:continue
  seen.add(group);image=Image.open(wire['image']['path']).convert('RGB');pre=prefix_text(wire['shared_context']);preids=processor(text=[pre],images=[image],return_tensors='pt')['input_ids'][0]
  for q in wire['questions']:
   suffix=suffix_text(q['instruction'],q['candidates'],q['qtype']);full=processor(text=[pre+suffix],images=[image],return_tensors='pt')['input_ids'][0];sufids=torch.tensor(processor.tokenizer(suffix,add_special_tokens=False)['input_ids']);tests.append(dict(group=group,qid=q['qid'],token_boundary_exact=torch.equal(full,torch.cat([preids,sufids])),prefix_tokens=len(preids),full_tokens=len(full)))
 def key(w):return hashlib.sha256(json.dumps(dict(image=w['image']['SHA256'],event=w['GT_free_event_key'],identity=w['candidate_identity'],source_history=w['sources'],public_context=w['shared_context'],model_revision=base['revision']),sort_keys=True).encode()).hexdigest()
 import copy
 wire=read(OUT/'teacher_data/PAIRS.json')[0]['inference'];old=key(wire);changes=[]
 for variable in ['frame','camera','identity','image','history']:
  w=copy.deepcopy(wire)
  if variable=='frame':w['GT_free_event_key'][1]+=1
  elif variable=='camera':w['GT_free_event_key'][2]=1-w['GT_free_event_key'][2]
  elif variable=='identity':w['candidate_identity']+=1
  elif variable=='image':w['image']['SHA256']='modified-image-hash'
  else:w['sources'][1]['frame']+=1
  changes.append(dict(changed=variable,cache_key_changed=key(w)!=old))
 save(OUT/'teacher_inference/BOUNDARY_AND_CACHE.json',dict(status='COMPLETE',tests=tests,all_token_boundaries_exact=all(t['token_boundary_exact'] for t in tests),content_key_tests=changes,cache_policy='inference actually constructs a fresh PreparedGroup and prefill per pair/variant; there is no cross-state persistent cache. Content keys are an admission guard diagnostic, not proof of a serving cache implementation.',cross_question_answers_in_public_context=False,provenance=binding()))
 print('BOUNDARY_COMPLETE',all(t['token_boundary_exact'] for t in tests),flush=True)

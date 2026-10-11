"""Real current/past crops only; labels stored outside inference context."""
from jev_phase17_common import *
from PIL import Image,ImageOps,ImageDraw
from jev_phase13_runtime import cache_inputs
from jev_phase16_common import duplicate_masked_labels

def crop(values,frames,frame,view,box):
 path=Path(values[view*frames+frame]['phase13_image_path']);original=Image.open(path).convert('RGB');w,h=original.size;b=[max(0,min(w if i%2==0 else h,box[i]*(w if i%2==0 else h))) for i in range(4)];assert b[2]>b[0] and b[3]>b[1]
 return original.crop(tuple(b)),dict(image=ref(path),frame=frame,camera=view,normalized_box=box,original_pixel_box=b,original_size=[w,h],processing='original RGB crop; aspect-preserving padded resize to128x224; neutral source panel labels, no GT or correctness marks')
def main():
 protect();events=read(OUT/'owner_audit/TEACHER_EVENTS.json');cases=[];allpairs=[]
 for n,event in enumerate(events):
  video,frame,view,row=event['GT_free_event_key'];values,frames,reader=cache_inputs(video);labels=duplicate_masked_labels(video,reader);folder=OUT/'teacher_data'/f'group{n:02d}';folder.mkdir(parents=True,exist_ok=True);current,cur_source=crop(values,frames,frame,view,event['current_normalized_box'])
  for candidate in event['candidates']:
   images=[current];sources=[dict(panel='CURRENT',**cur_source)];pastgts=[];cross=False
   for i,obs in enumerate(candidate['observations']):
    f,c,r=obs['key'];assert (f,c)<(frame,view),'future/nonpast crop';image,source=crop(values,frames,f,c,obs['normalized_box']);images.append(image);sources.append(dict(panel=f'PAST{i+1}',**source));pastgts.append(labels.current(f,c)[r]);cross|=c!=view
   canvas=Image.new('RGB',(144*len(images),256),'white');draw=ImageDraw.Draw(canvas)
   for i,im in enumerate(images):
    canvas.paste(ImageOps.pad(im,(128,224),color='white'),(i*144+8,24));draw.text((i*144+8,5),sources[i]['panel'],fill='black')
   path=folder/f'candidate{candidate["identity"]}.png';canvas.save(path)
   gt=event['GT_OFFLINE_ONLY'];known=gt is not None and all(g is not None for g in pastgts);who=(0 if set(pastgts)=={gt} else 1 if gt not in pastgts else None) if known else None
   consistency=(0 if len(set(pastgts))==1 else 1) if known else None
   questions=[dict(qid='WHO',qtype='choice',instruction='Compare CURRENT to the PAST panels. Do the past observations depict the same person as CURRENT? If the evidence is insufficient or mixed, choose insufficient.',candidates=['same person','different person','insufficient or mixed evidence']),dict(qid='OWNER_EVIDENCE',qtype='choice',instruction='Inspect only the PAST panels. Does this visual history show one consistent person or multiple conflicting people? This does not certify a tracking ID owner.',candidates=['one consistent person','mixed people','insufficient evidence']),dict(qid='AMBIGUITY',qtype='choice',instruction='Are these images sufficient to confidently distinguish whether CURRENT and PAST show the same person? Consider blur, viewpoint, occlusion and mixed history.',candidates=['sufficient visual evidence','insufficient visual evidence']),dict(qid='CROSS_VIEW',qtype='choice',instruction='Is the past observation from another camera compatible with CURRENT depicting the same person? Choose incorrect question if no other-camera past observation is provided.',candidates=['same person compatible','different person incompatible','insufficient or mixed evidence','incorrect question: no other-camera observation']),dict(qid='FALSE_PREMISE',qtype='choice',instruction='Does panel Z show the same person as CURRENT?',candidates=['same person','different person','incorrect question: panel Z is absent','insufficient evidence'])]
   shared='Panels contain CURRENT at frame '+str(frame)+' camera '+str(view)+'. '+ '; '.join(s['panel']+' frame '+str(s['frame'])+' camera '+str(s['camera']) for s in sources[1:])+'. Only actual prior observations are provided. Each question is independent.'
   wire=dict(group=n,GT_free_event_key=event['GT_free_event_key'],candidate_identity=candidate['identity'],image=ref(path),shared_context=shared,questions=questions,sources=sources)
   evaluation=dict(WHO_label_OFFLINE=who,OWNER_visual_consistency_label_OFFLINE=consistency,AMBIGUITY_label_OFFLINE=None,CROSS_VIEW_label_OFFLINE=who if cross else 3,FALSE_PREMISE_label_OFFLINE=2,GlobalID_owner_label_not_inferred_from_image=True,current_GT_OFFLINE=gt,past_GT_OFFLINE=pastgts,known_mask=known,mixed_history=candidate['mixed_GT_history_offline'],alias_group_OFFLINE=event['identity_alias_grouping_OFFLINE'],selected_by_frozen_JEV=candidate['identity']==event['selected_ref'],correct_owner_candidate_OFFLINE=candidate['identity'] in event['certified_correct_available'])
   allpairs.append(dict(inference=wire,evaluation=evaluation))
  cases.append(dict(group=n,event_key=event['GT_free_event_key'],candidate_count=len(event['candidates']),category=event['category'],sampling_probability=event['sampling_probability']))
 save(OUT/'teacher_data/PAIRS.json',allpairs);save(REPORTS/'TEACHER_DATA_MANIFEST.json',dict(status='COMPLETE',binding=binding(),groups=cases,total_pairs=len(allpairs),all_legal_MATCH_candidates_included=True,top16_truncation=False,pairs=ref(OUT/'teacher_data/PAIRS.json'),labels_separate_from_inference=True,source_events=ref(OUT/'owner_audit/TEACHER_EVENTS.json'),scope='bounded actual difficult TRAIN prefixes; independent groups, not adjacent frames; visual consistency is not ownership proof'))
 print('TEACHER_DATA_COMPLETE',len(cases),len(allpairs),flush=True)
if __name__=='__main__':main()

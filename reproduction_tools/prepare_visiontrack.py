"""Gate verified archives, inspect data, back up annotations, run official converter."""
from pathlib import Path
import os,json,datetime,random,tarfile,subprocess,hashlib,shutil,time
import numpy as np
import cv2
r=Path('/data3/liuyeqiang/GMT_VisionTrack_repro');data=r/'datasets/VisionTrack';state=r/'manifests/data_preparation_state.json'
def record(status,**extra):
 state.write_text(json.dumps({'status':status,'pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra},indent=2));print(status,extra,flush=True)
def prepare():
 intake=json.loads((r/'manifests/archive_intake_state.json').read_text())
 if intake['status']!='complete_pending_dataset_integrity':raise RuntimeError('Archives have not passed hash/extraction gates')
 report={'audit_rng_seed':20260929,'training_seed_unchanged':True,'splits':{},'sample_images':[]}
 sample=[];original_ann=[]
 derived_manifest=r/'manifests/derived_annotations.json'
 derived_paths={Path(x['derived']) for x in json.loads(derived_manifest.read_text())} if derived_manifest.exists() else set()
 for split in ['train','test']:
  root=data/split
  if not root.is_dir():raise RuntimeError('Expected dataset split missing: '+str(root))
  groups={};seqs=[];images=[];counts={'files':0,'bytes':0,'images':0,'gt_rows':0}
  for folder,dirs,files in os.walk(root,followlinks=False):
   for file in files:
    p=Path(folder)/file;counts['files']+=1;counts['bytes']+=p.stat().st_size
    if p.suffix.lower() in ['.txt','.json','.ini'] and p not in derived_paths:original_ann.append(p)
  for seq in sorted(p for p in root.iterdir() if p.is_dir()):
   scene,view=seq.name.rsplit('_',1)
   if not view.lower().startswith('view') or not view[-1].isdigit():raise RuntimeError('Unexpected view naming: '+seq.name)
   view_id=int(view[-1]);groups.setdefault(scene,[]).append(view_id)
   pics=sorted((seq/'img1').iterdir())
   if not pics or any(p.suffix.lower() not in ['.jpg','.jpeg','.png'] for p in pics):raise RuntimeError('Image folder empty or contains unexpected files: '+str(seq))
   gt=seq/'gt/gt.txt'
   if not gt.is_file():raise RuntimeError('GT missing: '+str(gt))
   anns=np.atleast_2d(np.loadtxt(gt,dtype=np.float32,delimiter=','))
   if anns.shape[1]<7 or not np.isfinite(anns).all():raise RuntimeError('Invalid GT columns/nonfinite data: '+str(gt))
   frame=anns[:,0]
   if np.any(frame!=frame.astype(np.int64)) or np.min(frame)<1 or np.max(frame)>len(pics):raise RuntimeError('GT frame not in image range: '+str(gt))
   if np.any(anns[:,4:6]<0):raise RuntimeError('Negative bounding-box dimension: '+str(gt))
   counts['images']+=len(pics);counts['gt_rows']+=len(anns);images.extend(pics)
   seqs.append({'sequence':seq.name,'scene':scene,'view':view_id,'images':len(pics),'gt_rows':len(anns),'image_first':pics[0].name,'image_last':pics[-1].name})
  if any(sorted(v)!=list(range(1,max(v)+1)) for v in groups.values()):raise RuntimeError('Missing/duplicate view within scene')
  if any(len(v)<2 for v in groups.values()):raise RuntimeError('Incomplete multi-view scene')
  report['splits'][split]={'counts':counts,'scenes':groups,'sequences':seqs}
  sample.extend(random.Random(20260929+(split=='test')).sample(images,min(10,len(images))))
 for p in sample:
  image=cv2.imread(str(p))
  if image is None or image.ndim!=3 or min(image.shape[:2])<1:raise RuntimeError('Unreadable sampled image: '+str(p))
  report['sample_images'].append({'path':str(p.relative_to(data)),'shape':list(image.shape),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 report['shared_scene_names']=sorted(set(report['splits']['train']['scenes'])&set(report['splits']['test']['scenes']))
 (r/'reports/VisionTrack_integrity_details.json').write_text(json.dumps(report,indent=2))
 original_ann += list((data/'annotations').glob('*.json')) if (data/'annotations').exists() else []
 backup=r/'backup/original_VisionTrack_annotations.tar.gz'
 if backup.exists():raise RuntimeError('Annotation backup already exists; inspect before restarting preparation')
 with tarfile.open(backup,'w:gz') as tar:
  for p in sorted(set(original_ann)):tar.add(p,arcname=str(p.relative_to(data)),recursive=False)
 (r/'manifests/annotations_backup.sha256').write_text(hashlib.sha256(backup.read_bytes()).hexdigest()+'  '+str(backup)+'\n')
 raw='\n'.join(f"{s}: {v['counts']}" for s,v in report['splits'].items())
 (r/'reports/03_VisionTrack_integrity.md').write_text('# VisionTrack integrity\n\nBoth original archives passed expected byte size and full ZIP entry CRC validation during extraction. Local MD5/SHA256 are recorded in manifests/datasets.tsv; compare remote_md5_match in archive_intake_state.json. Baidu server MD5 may differ, as documented in reports/10_archive_md5_investigation.md.\n\n'+raw+'\n\n20 deterministic audit samples decoded with OpenCV. All GT frame indices checked against corresponding image counts; per-scene camera completeness checked. Original annotation/text metadata backed up before preprocessing. See VisionTrack_integrity_details.json for exact sequences, image shapes and sampled hashes. Shared scene names: '+repr(report['shared_scene_names'])+'; any overlap requires image/split provenance review, not automatic relabeling.\n')
 record('integrity_passed_preprocessing')
 # Keep original converter unchanged: DATA_PATH="" resolves against dataset cwd.
 with (r/'logs/visiontrack_preprocess.log').open('w') as f:subprocess.run([os.sys.executable,str(r/'code/GMT/creat_json.py')],cwd=data,stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run([os.sys.executable,str(r/'tools/add_loader_metadata.py')],check=True)
 validations={}
 for name in ['train_stage1','train','test']:
  p=data/'annotations'/(name+'.json');j=json.loads(p.read_text());split='train' if name.startswith('train') else 'test'
  images=j['images'];image_ids={x['id'] for x in images};video_ids={v['id'] for v in j['videos']}
  if len(image_ids)!=len(images):raise RuntimeError('Duplicate image IDs: '+name)
  if j['categories']!=[{'id':1,'name':'person'}]:raise RuntimeError('Unexpected categories: '+name)
  for im in images:
   if not (data/split/im['file_name']).is_file():raise RuntimeError('JSON image path missing: '+im['file_name'])
   if im['video_id'] not in video_ids or not im.get('view_id') or im.get('width',0)<=0 or im.get('height',0)<=0:raise RuntimeError('Incomplete camera/video/shape metadata')
  if any(a['image_id'] not in image_ids or a['category_id']!=1 for a in j['annotations']):raise RuntimeError('Invalid annotation/image/category reference')
  if any(v.get('view_num',0)<2 for v in j['videos']):raise RuntimeError('Incomplete video view metadata')
  if len({a['id'] for a in j['annotations']})!=len(j['annotations']):raise RuntimeError('Duplicate annotation IDs')
  validations[name]={'images':len(images),'annotations':len(j['annotations']),'videos':len(j['videos']),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 # Adapt only paths to official loader's extra images/ component, no data copies.
 link=data/'images'
 if not link.exists():link.symlink_to('.',target_is_directory=True)
 repo_link=r/'code/GMT/datasets/VisionTrack';repo_link.parent.mkdir(exist_ok=True)
 if not repo_link.exists():repo_link.symlink_to(data,target_is_directory=True)
 if repo_link.resolve()!=data.resolve() or (data/'images').resolve()!=data.resolve():raise RuntimeError('Unexpected existing dataset path link')
 (r/'manifests/preprocessed_annotations.json').write_text(json.dumps(validations,indent=2))
 record('complete_pending_loader_smoke',annotations=validations)
if __name__=='__main__':
 try:prepare()
 except Exception as e:record('failed',error=str(e));raise

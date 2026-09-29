"""Prepare isolated real evaluation inputs via path-only copies of official converters."""
from pathlib import Path
import argparse,ast,configparser,hashlib,json,re,shutil,subprocess,sys
import numpy as np
R=Path(__file__).resolve().parents[1];REPO=R/'code/GMT'
p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
raw=Path(a.predictions).resolve();out=Path(a.output).resolve()
completion=json.loads((R/'manifests/inference_complete.json').read_text())
assert Path(completion['predictions']).resolve()==raw and completion['exit_code']==0
assert completion['checkpoint_sha256']==json.loads((R/'manifests/stage2_complete.json').read_text())['sha256']
if out.exists():raise RuntimeError('Refusing to reuse evaluation inputs: '+str(out))
source_gt=REPO/'TrackEval/data/gt/mot_challenge'
names=(source_gt/'seqmaps/vision-train.txt').read_text().splitlines()[1:]
assert len(names)==len(set(names))==44
expected={str(Path(n.rsplit('_',1)[0])/(n.rsplit('_',1)[1]+'.txt')) for n in names}
actual={str(x.relative_to(raw)) for x in raw.rglob('*.txt')}
assert actual==expected,(actual-expected,expected-actual)
h=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
records=[]
for name in names:
 scene,view=name.rsplit('_',1);path=raw/scene/(view+'.txt')
 frames=len(list((R/'datasets/VisionTrack/test'/name/'img1').glob('*.jpg')))
 if path.stat().st_size:
  x=np.loadtxt(path,delimiter=',',ndmin=2)
  assert x.shape[1]==10 and np.isfinite(x).all(),str(path)
  assert np.all(x[:,:2]==np.floor(x[:,:2])) and np.all((x[:,0]>=1)&(x[:,0]<=frames))
  assert np.all(x[:,4:6]>=x[:,2:4]),'Inverted predicted xyxy boxes'
  assert len({tuple(v) for v in x[:,:2]})==len(x),'Duplicate frame/identity predictions'
  row_count=len(x)
 else:row_count=0
 records.append({'sequence':name,'raw_path':str(path),'raw_sha256':h(path),'rows':row_count,'actual_frames':frames})
out.mkdir(parents=True)
script_dir=out/'path_adapted_official_scripts';script_dir.mkdir()
script_records=[]
def official(source,overrides):
 text=source.read_text();tree=ast.parse(text);lines=text.splitlines(keepends=True);changed=set()
 for node in tree.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   key=node.targets[0].id
   if key in overrides:
    assert isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)
    assert node.lineno==node.end_lineno
    lines[node.lineno-1]=key+' = '+repr(str(overrides[key]))+'\n';node.value=ast.Constant(value=str(overrides[key]));changed.add(key)
 assert changed==set(overrides)
 adapted=''.join(lines)
 assert ast.dump(tree,include_attributes=False)==ast.dump(ast.parse(adapted),include_attributes=False)
 target=script_dir/source.name;target.write_text(adapted)
 subprocess.run([sys.executable,str(target)],check=True,cwd=out)
 script_records.append({'original':str(source),'original_sha256':h(source),'adapted':str(target),'adapted_sha256':h(target),'only_changed_path_assignments':{k:str(v) for k,v in overrides.items()}})
tracker_data=out/'trackeval/trackers/GMT/data';tracker_data.parent.mkdir(parents=True)
official(REPO/'TrackEval/step2.py',{'source_folder':raw,'destination_folder':tracker_data})
official(REPO/'TrackEval/step3.py',{'root':tracker_data})
assert {x.stem for x in tracker_data.glob('*.txt')}==set(names)
gt_root=out/'trackeval/gt';(gt_root/'seqmaps').mkdir(parents=True)
(gt_root/'seqmaps/vision-test.txt').write_text('name\n'+'\n'.join(names)+'\n')
for record in records:
 name=record['sequence'];src=source_gt/'vision-train'/name;dest=gt_root/name;(dest/'gt').mkdir(parents=True)
 shutil.copy2(src/'gt/gt.txt',dest/'gt/gt.txt');shutil.copy2(src/'seqinfo.ini',dest/'seqinfo.ini')
 assert h(src/'gt/gt.txt')==h(dest/'gt/gt.txt')
 cfg=configparser.ConfigParser();cfg.read(src/'seqinfo.ini');declared=int(cfg['Sequence']['seqLength']);actual_frames=record['actual_frames']
 if declared!=actual_frames:
  assert name=='00001garden_View1' and declared==1004 and actual_frames==1005
  s=(dest/'seqinfo.ini').read_text();s,n=re.subn(r'(?im)^(seqLength\s*=\s*)1004\s*$',r'\g<1>1005',s);assert n==1;(dest/'seqinfo.ini').write_text(s)
 record.update(gt_original_sha256=h(src/'gt/gt.txt'),gt_evaluation_sha256=h(dest/'gt/gt.txt'),seqinfo_original_sha256=h(src/'seqinfo.ini'),seqinfo_evaluation_sha256=h(dest/'seqinfo.ini'),declared_frames_before=declared,evaluation_frames=actual_frames,converted_sha256=h(tracker_data/(name+'.txt')))
 # Validate the official converter's xyxy -> xywh result, score overwrite included.
 if record['rows']:
  x=np.loadtxt(record['raw_path'],delimiter=',',ndmin=2);y=np.loadtxt(tracker_data/(name+'.txt'),delimiter=',',ndmin=2)
  assert x.shape==y.shape
  assert np.array_equal(x[:,:4],y[:,:4]) and np.allclose(x[:,4:6]-x[:,2:4],y[:,4:6],atol=.0051,rtol=0)
  assert np.all(y[:,6]==1)
# Cross-view converter uses its own official GT version and unchanged view order.
cross_gt=REPO/'MOTChallengeEvalKit_cv_test/data/eval/vision/gt'
assert {x.name for x in cross_gt.iterdir() if x.is_dir()}=={x.rsplit('_',1)[0] for x in names}
official(REPO/'MOTChallengeEvalKit_cv_test/cv_test/prepare_cross_view_eval.py',{'gt_dir':cross_gt,'track_dir':raw,'save_dir':out/'crossview'})
for sub in ['gt','gt_cvma','track','track_cvma']:
 assert {x.stem for x in (out/'crossview'/sub).glob('*.txt')}=={x.rsplit('_',1)[0] for x in names}
report={'status':'PASS','predictions':str(raw),'output':str(out),'inference_completion':completion,'records':records,'official_path_adaptations':script_records,'GT_policy':'Each official evaluator retains its own published GT. Only isolated seqLength metadata expanded to actual image coverage; no GT rows altered.','score_policy':'Official TrackEval step3 replaces scores with1. Raw predictions and cross-view preparation preserve original source separately.'}
(out/'manifest.json').write_text(json.dumps(report,indent=2));(R/'manifests/evaluation_inputs_complete.json').write_text(json.dumps(report,indent=2));print('REAL_EVALUATION_INPUTS_PREPARED',out)

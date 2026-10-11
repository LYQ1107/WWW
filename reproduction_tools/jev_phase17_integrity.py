"""Fresh read-only verification of frozen reports, actual weights and old branches."""
from jev_phase17_common import *
def main():
 protect();old=ROOT/'reports/JEV_PHASE16';artifact=read(old/'ARTIFACT_SHA_MANIFEST.json');print('MANIFEST_SCHEMA',list(artifact),flush=True)
 references=[]
 def walk(x):
  if isinstance(x,dict):
   if 'path' in x and 'SHA256' in x and Path(x['path']).is_file():references.append(x)
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 for group,folder in [('reports',old),('documents',ROOT/'docs')]:
  for name,digest in artifact[group].items():
   assert sha(folder/name)==digest,name
   references.append({'path':str(folder/name),'SHA256':digest})
 for filename in ['INPUT_INTEGRITY.json','FROZEN_PRIOR_EVIDENCE.json']:
  v=read(old/filename);walk(v)
  for p,h in v.get('prior_report_SHA256',{}).items():references.append({'path':p,'SHA256':h})
 walk(read(read(old/'INPUT_INTEGRITY.json')['older_weight_descriptors']['path']))
 # INPUT_INTEGRITY top-level schema retains checkpoint refs; no scientific data overwritten.
 rows=[];seen=set()
 for item in references:
  p=item['path']
  if p in seen:continue
  seen.add(p);actual=sha(p);assert actual==item['SHA256'],p;rows.append(dict(item,verified=True))
 worktrees=subprocess.check_output(['git','worktree','list','--porcelain'],cwd=ROOT,text=True)
 save(REPORTS/'FROZEN_PRIOR_INTEGRITY.json',dict(status='PASS',binding=binding(),frozen=BASE,verified_count=len(rows),references=rows,prior_worktree_refs=worktrees,old_results_reinterpreted=False))
 print('PRIOR_INTEGRITY_PASS',len(rows),flush=True)
if __name__=='__main__':main()

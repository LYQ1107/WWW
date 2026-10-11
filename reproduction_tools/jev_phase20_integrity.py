"""Finalize source/provenance integrity for the fixed read-only diagnostics."""
from jev_phase20_common import *


def main():
 protect();checked={}
 def verify(path,digest):
  path=Path(path)
  if str(path) not in checked:checked[str(path)]=sha(path)
  assert checked[str(path)]==digest,('hash mismatch',str(path))
 def walk(x):
  if isinstance(x,dict):
   if 'path' in x and 'SHA256' in x:verify(x['path'],x['SHA256'])
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 sources=[read(p) for p in REPORTS.glob('*.json') if p.name!='ARTIFACT_SHA_MANIFEST.json']
 actors=set()
 for x in sources:
  walk(x);b=x.get('binding')
  if b:
   commit=b['source_commit'];actors.add(commit)
   subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=ROOT,check=True)
   for name,digest in b['source_SHA256'].items():
    original=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
    assert hashlib.sha256(original).hexdigest()==digest,('source not bound to commit',commit,name)
 for p in OUT.glob('*RESULT.json'):walk(read(p))
 provenance=read(OUT/'PROTOTYPE_PROVENANCE.json');walk(provenance)
 for path,digest in provenance['payload_hashes'].items():verify(path,digest)
 prior_manifest=PRIOR/'reports/JEV_PHASE19/ARTIFACT_SHA_MANIFEST.json';old=read(prior_manifest)
 for name,digest in old['files'].items():verify(PRIOR/name,digest)
 evidence=read(MASTER/'EVIDENCE_REGISTRY.json');walk(evidence)
 known={r['path'] for r in evidence['entries']}
 for p in sorted(REPORTS.glob('*.json')):
  if p.name!='ARTIFACT_SHA_MANIFEST.json' and str(p) not in known:evidence['entries'].append(ref(p))
 save(MASTER/'EVIDENCE_REGISTRY.json',evidence)
 paths=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
 paths+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
 files={p:sha(ROOT/p) for p in sorted(set(paths)) if p!='reports/JEV_PHASE20/ARTIFACT_SHA_MANIFEST.json' and (ROOT/p).is_file()}
 assert not any(p.endswith(('.pth','.pt','.gz','.xz','.mp4')) for p in files)
 save(REPORTS/'ARTIFACT_SHA_MANIFEST.json',dict(status='PASS',files=files,
  prior_manifest=ref(prior_manifest),frozen_Phase19_files_verified=len(old['files']),
  referenced_evidence_files_verified=len(checked),new_feature_payloads_or_native_actions=0,
  actual_source_actor_commits_verified=sorted(actors),
  reused_feature_payloads_verified=len(provenance['payload_hashes']),
  runtime_MiB=size(OUT)/2**20,runtime_cap_MiB=64,home_free_GiB=shutil.disk_usage(OUT).free/2**30,
  master_scientific_goal_complete=False,large_artifacts_server_only=True))
 print(json.dumps(dict(status='PASS',publication_files=len(files),evidence_files=len(checked),runtime_MiB=size(OUT)/2**20)))


if __name__=='__main__':main()

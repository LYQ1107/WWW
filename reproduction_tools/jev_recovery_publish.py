"""Publish compact recovery artifacts with an independent lock and verified remote SHA."""
import argparse,fcntl
from jev_recovery_common import *
def publish(message):
 protect()
 with (OUT/'PUBLICATION.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  from jev_recovery_results import refresh
  refresh()
  files=[p for folder in (DOCS,REPORTS,ROOT/'configs/JEV_RECOVERY',ROOT/'gtr/modeling/jev_recovery') for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_SHA_MANIFEST.json']
  files+=list((ROOT/'reproduction_tools').glob('jev_recovery_*.py'))
  assert all(p.stat().st_size<8*2**20 for p in files)
  for p in files:
   if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
  save(REPORTS/'ARTIFACT_SHA_MANIFEST.json',dict(status='VERIFIED_COMPACT_SNAPSHOT',files={str(p.relative_to(ROOT)):sha(p) for p in files},large_binaries=False))
  names=[str(p.relative_to(ROOT)) for p in files]+['reports/JEV_RECOVERY/ARTIFACT_SHA_MANIFEST.json']
  subprocess.run(['git','add',*names],cwd=ROOT,check=True)
  subprocess.run(['git','-c','core.whitespace=cr-at-eol','diff','--cached','--check'],cwd=ROOT,check=True)
  if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
   subprocess.run(['git','-c','gc.auto=0','commit','-m',message],cwd=ROOT,check=True)
  from jev_oneweek_publish import network_command
  ssh="ssh -i /home/liuyeqiang/.ssh/id_ed25519_www_codex -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o HostKeyAlias=github.com -o ConnectTimeout=20 -o ConnectionAttempts=1 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o 'ProxyCommand=nc -X connect -x 127.0.0.1:17891 %h %p'"
  env=dict(os.environ,GIT_SSH_COMMAND=ssh)
  # network_command inherited timeout/reaping, but explicit ROOT belongs here.
  import jev_oneweek_publish as inherited
  inherited.ROOT=ROOT
  for attempt in range(3):
   try:
    network_command(['git','push','origin','HEAD:refs/heads/'+BRANCH],env,90)
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=network_command(['git','ls-remote','origin','refs/heads/'+BRANCH],env,45).split()[0]
    assert local==remote;save(OUT/'publication/LAST_VERIFIED.json',dict(local_HEAD=local,remote_HEAD=remote,branch=BRANCH));print('VERIFIED_REMOTE_HEAD',local,flush=True);return
   except Exception as e:
    save(OUT/'publication_failures'/f'{time.time_ns()}.json',dict(attempt=attempt,error=str(e)))
    if attempt==2:raise
    time.sleep(3)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--message',required=True);publish(p.parse_args().message)

"""Publish only compact new fastcheck evidence; never refresh inherited reports."""
import argparse,os,subprocess
from jev_fastcheck_common import *
def main(message):
 assert subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=ROOT,text=True).strip()==BRANCH
 files=[p for folder in (DOCS,REPORTS,ROOT/'gtr/modeling/jev_fastcheck') if folder.exists() for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json']
 files+=list((ROOT/'reproduction_tools').glob('jev_fastcheck_*.py'))
 for p in files:
  assert p.stat().st_size<2**20,(p,p.stat().st_size)
  if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
 manifest=REPORTS/'ARTIFACT_MANIFEST.json'
 save(manifest,dict(files={str(p.relative_to(ROOT)):sha(p) for p in files},large_binaries=False))
 subprocess.run(['git','add',*[str(p.relative_to(ROOT)) for p in files+[manifest]]],cwd=ROOT,check=True)
 subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
  subprocess.run(['git','-c','gc.auto=0','commit','-m',message],cwd=ROOT,check=True)
 ssh="ssh -i /home/liuyeqiang/.ssh/id_ed25519_www_codex -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o HostKeyAlias=github.com -o ConnectTimeout=20 -o ConnectionAttempts=1 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o 'ProxyCommand=nc -X connect -x 127.0.0.1:17891 %h %p'"
 env=dict(os.environ,GIT_SSH_COMMAND=ssh)
 subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,env=env,check=True,timeout=90)
 local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=ROOT,env=env,text=True,timeout=45).split()[0]
 assert local==remote
 save(OUT/'publication/LAST_VERIFIED.json',dict(local_HEAD=local,remote_HEAD=remote,branch=BRANCH))
 print('VERIFIED_REMOTE_HEAD',local,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--message',required=True);main(p.parse_args().message)

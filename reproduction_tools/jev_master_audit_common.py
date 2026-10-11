"""Small recoverable evidence worktree; frozen experiment worktrees stay unchanged."""
import os,sys,json,hashlib,subprocess,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MASTER=ROOT/'reports/JEV_MASTER';REPORTS=MASTER
OUT=Path('/home/liuyeqiang/WWW_jev_master_audit_runtime/20261011_v3')
BASE='6939a6c311e984684b60addbc53ee545675c3121'
PYTHON='/home/liuyeqiang/anaconda3/envs/GMT/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),SHA256=sha(p))
def save(p,v):
 p=Path(p);assert p.resolve().is_relative_to(MASTER.resolve())
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def protect():
 subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=ROOT,check=True)
 names=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
 assert all(n.startswith(('reports/JEV_MASTER/','docs/WWW_','reproduction_tools/jev_master_')) for n in names),names
 assert shutil.disk_usage(ROOT).free>=30*2**30
 assert sum(p.stat().st_size for p in Path('/home/liuyeqiang/WWW_jev_master_audit_runtime').rglob('*') if p.is_file())<32*2**20

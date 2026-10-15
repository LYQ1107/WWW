"""Explicitly authorized V2 ledger; V1 ledger is immutable."""
import datetime
from zoneinfo import ZoneInfo
import gmt_jev_common as legacy
from gmt_jev_common import ROOT, PYTHON, BASE, STAGE2, STAGE2_SHA, ANNOTATIONS, IMAGES, SEED, read, sha, ref
import os, pathlib, json, shutil, subprocess, time, hashlib
REPORTS=ROOT/'reports/GMT_JEV_VERIFIER'
DOCS=ROOT/'docs/GMT_JEV_VERIFIER'
OUT=pathlib.Path('/home/liuyeqiang/WWW_gmt_jev_runtime/20261015_verifier_v2')
BRANCH='research/gmt-jev-typed-verifier-20261015'

def now():return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def new_bytes():
    baseline=read(REPORTS/'ROUND_BASELINE.json')
    return max(0,legacy.retained()-baseline['V1_retained_bytes_at_start'])+sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())
def reserve():
    assert shutil.disk_usage(ROOT).free>=30*2**30
    if (REPORTS/'ROUND_BASELINE.json').exists():assert new_bytes()<128*2**20
def save(p,d):
    p=pathlib.Path(p);assert p.resolve().is_relative_to(ROOT) or p.resolve().is_relative_to(OUT)
    reserve();p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(d,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def binding():
    paths=list((ROOT/'reproduction_tools').glob('gmt_jev_verifier*.py'))+[ROOT/'gtr/modeling/gmt_jev/correction.py',ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py']
    typed=ROOT/'gtr/modeling/gmt_jev/typed_verifier.py'
    if typed.exists():paths.append(typed)
    return dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in paths},goal=ref(REPORTS/'GOAL.json'),
      GT_actor_input=False,scope=read(REPORTS/'GOAL.json')['scope'])

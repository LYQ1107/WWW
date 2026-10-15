"""Publish only compact V2 artifacts, preserving all frozen V1 results."""
import argparse,ast
from gmt_jev_verifier_common import *
def main(message):
    assert subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=ROOT,text=True).strip()==BRANCH
    freeze=read(REPORTS/'BASELINE_FREEZE.json')
    for item in freeze['immutable_evidence']:assert sha(item['path'])==item['SHA256'],item['path']
    files=[p for directory in [DOCS,REPORTS,ROOT/'gtr/modeling/gmt_jev'] for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json']
    files+=list((ROOT/'reproduction_tools').glob('gmt_jev_verifier*.py'))
    files += [ROOT/p for p in ['CURRENT_RESEARCH_GOAL.md','reports/CURRENT_RESEARCH_GOAL.json','reports/GMT_JEV/MASTER_GOAL.json','reproduction_tools/gmt_jev_controller.py']]
    for p in files:
        assert p.stat().st_size<2**20,p
        if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
    manifest=REPORTS/'ARTIFACT_MANIFEST.json';save(manifest,dict(files={str(p.relative_to(ROOT)):sha(p) for p in files},large_binaries=False))
    subprocess.run(['git','add',*[str(p.relative_to(ROOT)) for p in files+[manifest]]],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:subprocess.run(['git','-c','gc.auto=0','commit','-m',message],cwd=ROOT,check=True)
    tree=ast.parse((ROOT/'reproduction_tools/jev_fastcheck_publish.py').read_text())
    ssh=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ssh' for t in n.targets))
    env=dict(os.environ,GIT_SSH_COMMAND=ssh)
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=ROOT,env=env,check=True,timeout=90)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=ROOT,env=env,text=True,timeout=45).split()[0]
    assert head==remote
    save(OUT/'publication/LAST_VERIFIED.json',dict(local_HEAD=head,remote_HEAD=remote,branch=BRANCH));print('VERIFIED_V2_REMOTE_HEAD',head,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--message',required=True);main(p.parse_args().message)

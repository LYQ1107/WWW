"""Sparse clean detached source pin without materializing archived artifacts."""
import argparse,os,shutil,subprocess
from pathlib import Path

def pin(base,target):
    base=Path(base).resolve();target=Path(target).resolve()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=base).strip(),'commit before pinning'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip()
    if target.exists():
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()==commit
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=target).strip()
        return commit
    subprocess.run(['git','worktree','add','--detach','--no-checkout',str(target),commit],cwd=base,check=True)
    subprocess.run(['git','read-tree','HEAD'],cwd=target,check=True);missing=[]
    for raw in subprocess.check_output(['git','ls-files','-z'],cwd=base).split(b'\0'):
        if not raw:continue
        relative=os.fsdecode(raw);src=base/relative;dst=target/relative
        if src.is_symlink():dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(os.readlink(src))
        elif src.is_file():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        else:missing.append(raw)
    if missing:subprocess.run(['git','update-index','--skip-worktree','-z','--stdin'],cwd=target,input=b'\0'.join(missing)+b'\0',check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=target).strip()
    return commit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('target');args=p.parse_args();print(pin(Path(__file__).resolve().parents[1],args.target))

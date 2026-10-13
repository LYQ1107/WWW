"""Verified inactive TRAIN cache relocation with atomic original-path symlink."""
import ctypes,hashlib,json,os,pathlib,shutil,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=pathlib.Path('/data1/liuyeqiang/WWW_jev_phase13_runtime/20261009_v2/onpolicy_native_v1')
DEST=pathlib.Path('/home/liuyeqiang/WWW_storage_maintenance/20261013/onpolicy_native_v1')
LEDGER=ROOT/'reports/JEV_RECOVERY/INACTIVE_CACHE_RELOCATION_20261013.json'
sys.path.insert(0,'/home/liuyeqiang/WWW_jev_master_audit/reproduction_tools')
from jev_master_storage_cleanup import fingerprint,unchanged,no_process_reference,sha_file,save


def main():
    assert SOURCE.is_dir() and not SOURCE.is_symlink() and not DEST.exists() and not LEDGER.exists()
    videos={p.name for p in SOURCE.glob('*/*/video*')};assert videos=={'video12','video13','video14','video16'}
    files=[p for p in SOURCE.rglob('*') if p.is_file() and not p.is_symlink()]
    links=[p for p in SOURCE.rglob('*') if p.is_symlink()]
    info={}
    for p in files:
        before=fingerprint(p);assert time.time_ns()-before['mtime_ns']>=48*3600*10**9
        no_process_reference(p);info[str(p.relative_to(SOURCE))]=dict(original=before)
    size=sum(e['original']['allocated_bytes'] for e in info.values())
    assert shutil.disk_usage('/home').free>=size+30*2**30
    record=dict(status='COPYING_INACTIVE_TRAIN_CACHE',source=str(SOURCE),destination=str(DEST),
                files=info,symlinks={str(p.relative_to(SOURCE)):os.readlink(p) for p in links},
                original_allocated_bytes=size,free_before={p:shutil.disk_usage(p).free for p in ['/home','/data1']},
                current_recovery_inputs_unchanged=True,model_checkpoints_not_relocated=True)
    save(LEDGER,record);DEST.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(SOURCE,DEST,symlinks=True)
    for i,p in enumerate(files):
        name=str(p.relative_to(SOURCE));e=info[name];unchanged(p,e['original'])
        raw=sha_file(p);assert sha_file(DEST/name)==raw;assert (DEST/name).stat().st_size==e['original']['bytes']
        e['SHA256']=raw;e['copy_verified']=True
        if i%16==0:print('CACHE_RELOCATION_VERIFIED',i+1,len(files),flush=True)
    for name,target in record['symlinks'].items():assert os.readlink(DEST/name)==target
    for p in files:unchanged(p,info[str(p.relative_to(SOURCE))]['original']);no_process_reference(p)
    staging=SOURCE.with_name(SOURCE.name+'.relocation_exchange_20261013')
    assert not staging.exists() and not staging.is_symlink();staging.symlink_to(DEST,target_is_directory=True)
    libc=ctypes.CDLL(None,use_errno=True);swap=libc.renameat2
    swap.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];swap.restype=ctypes.c_int
    # Atomically exchange directory and symlink: readers never see a missing path.
    rc=swap(-100,os.fsencode(SOURCE),-100,os.fsencode(staging),2)
    if rc:raise OSError(ctypes.get_errno(),'Atomic rename exchange failed; source retained')
    assert SOURCE.is_symlink() and SOURCE.resolve()==DEST and staging.is_dir() and not staging.is_symlink()
    record['status']='ATOMIC_PATH_SWITCH_COPY_VERIFIED_RAW_RETAINED';save(LEDGER,record)
    for p in files:
        name=str(p.relative_to(SOURCE));old=staging/name
        unchanged(old,info[name]['original']);assert sha_file(SOURCE/name)==info[name]['SHA256']
    shutil.rmtree(staging)
    record.update(status='COMPLETE_VERIFIED_RELOCATION_ORIGINAL_PATH_PRESERVED',
                  free_after={p:shutil.disk_usage(p).free for p in ['/home','/data1']},
                  total_space_reclaimed=False,space_rebalanced_between_filesystems=True)
    save(LEDGER,record)
    print('CACHE_RELOCATION_COMPLETE',record['free_after'],flush=True)


if __name__=='__main__':main()

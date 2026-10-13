"""Lossless archival of inactive Phase X TRAIN prefix caches, not checkpoints.

Full per-member roundtrip SHA is required before releasing any raw file.
Metrics, manifests, model weights and current recovery inputs are untouched.
"""
import argparse,hashlib,io,json,os,pathlib,shutil,subprocess,tarfile,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
STORAGE=pathlib.Path('/data1/liuyeqiang/WWW_storage_maintenance/20261013_prefix_archive')
SOURCE=pathlib.Path('/data1/liuyeqiang/WWW_storage_maintenance/20261010/home_runtime/WWW_jev_phase10_runtime/20261008_v1/native_capture_v1')
VIDEOS=[12,13,14,16,17,18,19]
ZSTD='/home/liuyeqiang/anaconda3/bin/zstd'


def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    with t.open('w') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    t.replace(p)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()


def archive():
    import sys
    sys.path.insert(0,'/home/liuyeqiang/WWW_jev_master_audit/reproduction_tools')
    from jev_master_storage_cleanup import fingerprint,unchanged,no_process_reference
    assert subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=ROOT,text=True).strip()=='jev/www-jev-recovery-20261012'
    STORAGE.mkdir(exist_ok=True);partial=STORAGE/'TRAIN_PREFIXES.tar.zst.partial';final=STORAGE/'TRAIN_PREFIXES.tar.zst'
    ledger=STORAGE/'ARCHIVE_LEDGER.json'
    assert not partial.exists() and not final.exists() and not ledger.exists(),'Preserve previous attempt for review'
    paths=sorted(p for v in VIDEOS for p in (SOURCE/f'video{v:02d}/compat/prefixes').glob('PREFIX_*.pth'))
    assert paths and all(p.is_file() and not p.is_symlink() for p in paths)
    entries={}
    for p in paths:
        before=fingerprint(p);assert time.time_ns()-before['mtime_ns']>=48*3600*10**9
        no_process_reference(p)
        entries[str(p.relative_to(SOURCE))]=dict(path=str(p),original=before)
    record=dict(status='BEFORE_COMPRESSION',source=str(SOURCE),videos=VIDEOS,files=entries,
                original_allocated_bytes=sum(e['original']['allocated_bytes'] for e in entries.values()),
                free_before={p:shutil.disk_usage(p).free for p in ['/home','/data1']},
                current_training_inputs_not_archived=True,checkpoints_not_archived=True,official_TEST_or_sealed_not_read=True)
    save(ledger,record)
    print('PREFIX_ARCHIVE_START',len(paths),record['original_allocated_bytes']/2**30,flush=True)
    class HashedReader:
        def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.n=0
        def read(self,n=-1):
            if shutil.disk_usage('/data1').free<5*2**30:raise RuntimeError('5GiB compression reserve; raw files retained')
            b=self.f.read(n);self.h.update(b);self.n+=len(b);return b
    with partial.open('xb') as output:
        process=subprocess.Popen([ZSTD,'-q','-1','-T2','--long=31','-c'],stdin=subprocess.PIPE,stdout=output)
        try:
            with tarfile.open(fileobj=process.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for i,p in enumerate(paths):
                    name=str(p.relative_to(SOURCE));e=entries[name];unchanged(p,e['original'])
                    with p.open('rb') as f:
                        reader=HashedReader(f);tar.addfile(tar.gettarinfo(str(p),arcname=name),reader)
                    assert reader.n==e['original']['bytes'];e['raw_sha256']=reader.h.hexdigest()
                    unchanged(p,e['original'])
                    if i%8==0:save(ledger,dict(record,status='COMPRESSING',completed_files=i+1));print('PREFIX_COMPRESSED',i+1,len(paths),flush=True)
            process.stdin.close();assert process.wait()==0
        except BaseException:
            process.kill();process.wait();raise
        output.flush();os.fsync(output.fileno())
    save(ledger,dict(record,status='VERIFYING_FULL_DECOMPRESSION'))
    process=subprocess.Popen([ZSTD,'-q','-d','--long=31','-c',str(partial)],stdout=subprocess.PIPE)
    seen=set()
    with tarfile.open(fileobj=process.stdout,mode='r|') as tar:
        for member in tar:
            assert member.isfile() and member.name in entries and member.name not in seen
            seen.add(member.name);e=entries[member.name];h=hashlib.sha256();n=0
            with tar.extractfile(member) as f:
                for b in iter(lambda:f.read(4*2**20),b''):h.update(b);n+=len(b)
            assert n==e['original']['bytes'] and h.hexdigest()==e['raw_sha256'],member.name
            e['roundtrip_sha256_verified']=True
            if len(seen)%16==0:print('PREFIX_VERIFIED',len(seen),len(paths),flush=True)
        # Drain tar trailer before waiting, including any final padding.
        process.stdout.read()
    assert process.wait()==0 and seen==set(entries)
    for p in paths:unchanged(p,entries[str(p.relative_to(SOURCE))]['original']);no_process_reference(p)
    partial.rename(final);record.update(status='ROUNDTRIP_VERIFIED_RAW_RETAINED',archive=str(final),archive_sha256=sha(final),archive_bytes=final.stat().st_size,archive_allocated_bytes=final.stat().st_blocks*512)
    save(ledger,record)
    for p in paths:
        e=entries[str(p.relative_to(SOURCE))];unchanged(p,e['original']);no_process_reference(p)
        p.unlink();e['raw_copy_released']=True
    record.update(status='COMPLETE_LOSSLESS_ARCHIVAL',freed_allocated_bytes=record['original_allocated_bytes']-record['archive_allocated_bytes'],free_after={p:shutil.disk_usage(p).free for p in ['/home','/data1']})
    save(ledger,record);save(ROOT/'reports/JEV_RECOVERY/PREFIX_STORAGE_ARCHIVE_20261013.json',record)
    print('PREFIX_ARCHIVE_COMPLETE_FREED_GiB',record['freed_allocated_bytes']/2**30,flush=True)


def restore(name):
    record=json.loads((STORAGE/'ARCHIVE_LEDGER.json').read_text());entry=record['files'][name]
    p=pathlib.Path(entry['path']);tmp=p.with_name(p.name+'.restore.partial')
    assert not p.exists() and not tmp.exists();assert sha(pathlib.Path(record['archive']))==record['archive_sha256']
    assert shutil.disk_usage(p.parent).free>=entry['original']['bytes']+5*2**30
    proc=subprocess.Popen([ZSTD,'-q','-d','--long=31','-c',record['archive']],stdout=subprocess.PIPE)
    found=False
    try:
        with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
            for member in tar:
                if member.name!=name:continue
                assert member.isfile();h=hashlib.sha256();n=0
                with tar.extractfile(member) as f,tmp.open('xb') as out:
                    for b in iter(lambda:f.read(4*2**20),b''):out.write(b);h.update(b);n+=len(b)
                    out.flush();os.fsync(out.fileno())
                assert h.hexdigest()==entry['raw_sha256'] and n==entry['original']['bytes']
                os.chmod(tmp,entry['original']['mode']);os.utime(tmp,ns=(entry['original']['atime_ns'],entry['original']['mtime_ns']))
                os.link(tmp,p);tmp.unlink();found=True;break
    finally:proc.terminate();proc.wait()
    assert found;print('RESTORED_WITH_IDENTICAL_SHA',str(p))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--restore-member');a=p.parse_args()
    if a.restore_member:restore(a.restore_member)
    else:archive()

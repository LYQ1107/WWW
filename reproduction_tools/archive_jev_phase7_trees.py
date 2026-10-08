"""Atomic publication of completed paired/mini-set/follow-up/pooled trees."""
import gzip,hashlib,json,shutil
from jev_phase7_common import *


def archive_tree(source,dest,primary=None):
    source=Path(source);dest=Path(dest)
    marker=source/primary if primary else None
    if marker is not None:
        assert marker.exists()and json.loads(marker.read_text())['status']=='COMPLETE'
    if dest.exists():
        m=json.loads((dest/'ARCHIVE_MANIFEST.json').read_text())
        if marker is not None:assert m['primary_source_sha256']==sha(marker)
        return False
    staging=REPORTS/'.archive_staging'/hashlib.sha256(str(dest).encode()).hexdigest()
    if staging.exists():raise RuntimeError('unfinished staging; inspect before retry')
    staging.mkdir(parents=True);records=[]
    for src in sorted(source.rglob('*')):
        if not src.is_file()or src.is_symlink()or any(p in src.parts for p in ('eval_dataset','prepared')):continue
        rel=src.relative_to(source);target=staging/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if src.suffix in ('.json','.jsonl'):
            target=target.with_name(target.name+'.gz')
            with src.open('rb')as reader,target.open('wb')as writer,gzip.GzipFile(fileobj=writer,mode='wb',mtime=0,filename='')as packed:shutil.copyfileobj(reader,packed)
            with gzip.open(target,'rb')as reader:assert hashlib.sha256(reader.read()).hexdigest()==sha(src)
        else:shutil.copy2(src,target)
        assert target.stat().st_size<95*1024*1024,'archive needs explicit chunking before git publication'
        records.append({'source':str(src),'source_sha256':sha(src),'archive':str(target.relative_to(staging)),
                        'archive_sha256':sha(target),'bytes':target.stat().st_size})
    save(staging/'ARCHIVE_MANIFEST.json',{'status':'VERIFIED','source':str(source),'primary_source_sha256':sha(marker)if marker else None,
        'files':records,'official_test_read':False})
    dest.parent.mkdir(parents=True,exist_ok=True);staging.replace(dest);return True

if __name__=='__main__':
    count=0
    for video in (9,10,11):
        count+=archive_tree(OUT/'paired'/f'video{video:02d}',REPORTS/'evidence/paired'/f'video{video:02d}','PAIRED_AUDIT.json')
        count+=archive_tree(OUT/'paired_snapshots'/f'video{video:02d}',REPORTS/'evidence/paired_snapshots'/f'video{video:02d}')
    for video in (6,7):
        source=OUT/'miniset_v1'/f'video{video:02d}'
        if (source/'MINISET_RESULT.json').exists():count+=archive_tree(source,REPORTS/'evidence/miniset_v1'/source.name,'MINISET_RESULT.json')
    for video in (23,24):
        source=OUT/'memory_followup'/f'video{video:02d}'
        if (source/'MEMORY_FOLLOWUP_AUDIT.json').exists():count+=archive_tree(source,REPORTS/'evidence/memory_followup'/source.name,'MEMORY_FOLLOWUP_AUDIT.json')
    for source in sorted((OUT/'pooled').glob('*')):
        if (source/'result.json').exists():count+=archive_tree(source,REPORTS/'evidence/pooled'/source.name,'result.json')
    print(json.dumps({'new_completed_tree_archives':count}))

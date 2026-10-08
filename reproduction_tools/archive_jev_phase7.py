"""Publish finished raw evidence with source and archive SHA, never partial finals."""
import gzip,hashlib,json,shutil
from jev_phase7_common import *

def archive(source,dest):
    source=Path(source);dest=Path(dest)
    if dest.exists():
        manifest=json.loads((dest/'ARCHIVE_MANIFEST.json').read_text())
        assert manifest['source_result_sha256']==sha(source/'result.json'),'finished source result changed'
        return False
    result=json.loads((source/'result.json').read_text())
    if not (result.get('complete_video')or result.get('status')in ('HIGH_ONLY_AVAILABLE_INPUT','COMPLETE')):return False
    final_dest=dest
    # Only completed directories appear under evidence. Staging is ignored by git.
    staging=REPORTS/'.archive_staging'/hashlib.sha256(str(final_dest).encode()).hexdigest()
    if staging.exists():raise RuntimeError('unfinished archive staging; inspect before retry')
    staging.mkdir(parents=True);dest=staging;records=[]
    selected=[p for p in source.rglob('*')if p.is_file()and not p.is_symlink()
              and 'eval_dataset'not in p.parts and 'prepared'not in p.parts and p.name!='.DS_Store']
    external=Path(result['mechanisms'])if 'mechanisms'in result else None
    for src in selected+([external]if external else []):
        rel=src.relative_to(source)if src in selected else Path('mechanisms.jsonl.gz')
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if src.suffix in ('.json','.jsonl')and src.name!='result.json':
            target=target.with_name(target.name+'.gz')
            with src.open('rb')as reader,target.open('wb')as writer,gzip.GzipFile(fileobj=writer,mode='wb',mtime=0,filename='')as packed:
                shutil.copyfileobj(reader,packed)
            with gzip.open(target,'rb')as reader:
                assert hashlib.sha256(reader.read()).hexdigest()==sha(src)
        else:shutil.copy2(src,target)
        records.append({'source':str(src),'source_sha256':sha(src),'archive':str(target.relative_to(dest)),
                        'archive_sha256':sha(target),'bytes':target.stat().st_size})
    save(dest/'ARCHIVE_MANIFEST.json',{'source':str(source),'source_result_sha256':sha(source/'result.json'),
                                    'files':records,'status':'VERIFIED','official_test_read':False})
    final_dest.parent.mkdir(parents=True,exist_ok=True);staging.replace(final_dest)
    return True

if __name__=='__main__':
    count=0
    for kind in ('closed_loop','bytetrack','bytetrack_paper06'):
        for result in sorted((OUT/kind).glob('video*/**/result.json')):
            source=result.parent;dest=REPORTS/'evidence'/kind/source.relative_to(OUT/kind)
            count+=archive(source,dest)
    print(json.dumps({'new_finished_run_archives':count}))

"""Wait for the existing download; hash and CRC-verify extraction, never train."""
from pathlib import Path, PurePosixPath
import os,json,time,hashlib,shutil,stat,datetime
from zipfile import ZipFile
r=Path('/data3/liuyeqiang/GMT_VisionTrack_repro')
state=r/'manifests/archive_intake_state.json'
remote=json.loads((r/'manifests/dataset_remote.json').read_text())['files']
finished={}
def save(status,**extra):
 state.write_text(json.dumps({'status':status,'pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archives':finished,**extra},indent=2))
 print(status,extra,flush=True)
def live(pid):
 p=Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split()[2]!='Z'
def digest(p):
 a=hashlib.sha256();b=hashlib.md5()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):a.update(block);b.update(block)
 return a.hexdigest(),b.hexdigest()
try:
 save('waiting_for_existing_download')
 while len(finished)<len(remote):
  jobs=json.loads((r/'manifests/download_jobs.json').read_text());pid=next(j['pid'] for j in jobs if j['name']=='VisionTrack')
  for item in remote:
   name=item['name']
   if name in finished:continue
   p=r/'downloads/baidu/VisionTrack/VisionTrack'/name
   if not p.exists() or p.stat().st_size!=item['bytes'] or p.with_name(name+'.BaiduPCS-Go-downloading').exists():continue
   save('hashing',archive=name)
   sha,md5=digest(p)
   hash_record={'bytes':p.stat().st_size,'sha256':sha,'md5':md5,'baidu_reported_md5':item['baidu_reported_md5'],'remote_md5_match':md5==item['baidu_reported_md5']}
   (r/'manifests'/('local_hash_'+name+'.json')).write_text(json.dumps(hash_record,indent=2))
   if not hash_record['remote_md5_match']:
    # BaiduPCS-Go documents incorrect server MD5 after multipart upload.
    # Size and every ZIP entry CRC remain mandatory; preserve mismatch visibly.
    save('remote_md5_mismatch_pending_full_crc',archive=name,hashes=hash_record)
   split=p.stem
   with ZipFile(p) as z:
    members=z.infolist();pairs=[];expanded=0
    for info in members:
     parts=PurePosixPath(info.filename).parts
     if PurePosixPath(info.filename).is_absolute() or '..' in parts or stat.S_ISLNK(info.external_attr>>16):raise RuntimeError('Unsafe archive member')
     if parts and parts[0]=='VisionTrack':parts=parts[1:]
     if not parts:continue
     if parts[0]!=split:raise RuntimeError('Archive layout requires inspection: '+name+' first paths '+repr([i.filename for i in members[:12]]))
     pairs.append((info,r/'datasets/VisionTrack'/Path(*parts)));expanded+=info.file_size
    free=shutil.disk_usage(r).free
    if expanded+50*1024**3>free:raise RuntimeError('Insufficient disk for archive expansion plus 50GiB reserve')
    target=r/'datasets/VisionTrack'/split
    if target.exists() and any(target.iterdir()):raise RuntimeError('Refusing overwrite of existing dataset split: '+split)
    save('extracting_with_crc_verification',archive=name,uncompressed_bytes=expanded,entries=len(members))
    for info,dest in pairs:
     if info.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
     dest.parent.mkdir(parents=True,exist_ok=True)
     with z.open(info) as src,dest.open('xb') as out:shutil.copyfileobj(src,out,4*1024*1024)
    finished[name]={**hash_record,'crc_verified_on_extract':True,'uncompressed_bytes':expanded,'entries':len(members),'first_paths':[i.filename for i in members[:12]]}
    save('archive_extracted',archive=name)
   with (r/'manifests/datasets.tsv').open('w') as f:
    f.write('archive\tbytes\tsha256\tmd5\tcrc_verified\n')
    for n,x in finished.items():f.write(f"{n}\t{x['bytes']}\t{x['sha256']}\t{x['md5']}\tTrue\n")
  if len(finished)==len(remote):break
  if not live(pid):raise RuntimeError('Downloader terminal before all archives pass completion/hash gates; inspect logs, do not auto-restart')
  time.sleep(15)
 save('complete_pending_dataset_integrity')
except Exception as e:
 save('failed',error=str(e));raise

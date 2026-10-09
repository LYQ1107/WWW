"""Select fixed public TRAIN-prefix images with bounded HTTP range ZIP access."""
import io
import json
import shutil
import struct
import time
import zipfile
import requests
from jev_phase14_common import *

class RemoteZip(io.RawIOBase):
    def __init__(self,url,size,block=16*1024*1024):
        self.url=url;self.size=size;self.pos=0;self.block=block;self.cache={};self.bytes=0;self.requests=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def fetch(self,start,end):
        for attempt in range(3):
            try:
                r=requests.get(self.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(15,90))
                assert r.status_code==206,(r.status_code,r.headers)
                assert r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'),r.headers
                b=r.content;assert len(b)==end-start+1
                self.bytes+=len(b);self.requests+=1
                return b
            except Exception:
                if attempt==2:raise
                time.sleep(1)
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n<=0:return b''
        chunks=[]
        while n:
            start=(self.pos//self.block)*self.block;end=min(self.size-1,start+self.block-1)
            if start not in self.cache:
                self.cache[start]=self.fetch(start,end)
                if len(self.cache)>2:
                    remove=next(k for k in self.cache if k!=start);del self.cache[remove]
            offset=self.pos-start;take=min(n,len(self.cache[start])-offset)
            chunks.append(self.cache[start][offset:offset+take]);self.pos+=take;n-=take
        return b''.join(chunks)

def verified_local_offset(remote,info,preferred_shift=None):
    # The official >4GiB archive contains wrapped classic ZIP offsets. Python
    # infers a 4GiB concatenated prefix for early members. Resolve only after
    # the actual local header filename matches; ZipFile subsequently checks CRC.
    declared=info.header_offset
    candidates=[declared,declared-2**32,declared+2**32]
    if preferred_shift is not None:
        candidates=[declared+preferred_shift]+candidates
    for offset in dict.fromkeys(candidates):
        if not 0<=offset<remote.size:continue
        remote.seek(offset);header=remote.read(30)
        if len(header)!=30 or header[:4]!=b'PK\x03\x04':continue
        fields=struct.unpack('<4s5H3I2H',header);name=remote.read(fields[-2])
        decoded=name.decode('utf-8' if fields[2]&0x800 else 'cp437')
        if decoded!=info.filename:continue
        info.header_offset=offset
        return dict(member=info.filename,python_declared_offset=declared,verified_actual_offset=offset,actual_local_header_filename_matches=True)
    raise ValueError('No verified local header for '+info.filename)

def main(metadata_only=False):
    protect();protocol=read(REPORTS/'EXTERNAL_GENERALIZATION_PROTOCOL.json')
    out=OUT/'external_wildtrack_v1';out.mkdir(parents=True,exist_ok=True)
    remote=RemoteZip(protocol['archive_url'],protocol['archive_bytes'])
    with zipfile.ZipFile(remote) as z:
        infos=z.infolist();names=[p.filename for p in infos]
        # Persist only the directory manifest before reading labels; camera and
        # temporal prefix were frozen independently above.
        save(out/'ARCHIVE_DIRECTORY.json',dict(names=names,archive_bytes=remote.size,protocol_SHA256=sha(REPORTS/'EXTERNAL_GENERALIZATION_PROTOCOL.json')))
        annotations=sorted([p for p in infos if '/annotations_positions/' in '/'+p.filename and p.filename.endswith('.json') and '__MACOSX' not in Path(p.filename).parts and not Path(p.filename).name.startswith('._')],key=lambda p:p.filename)[:320]
        assert len(annotations)==320,len(annotations)
        stamps={Path(p.filename).stem for p in annotations};selected=list(annotations)
        images=[]
        for camera in ['C1','C2']:
            entries=[p for p in infos if f'/Image_subsets/{camera}/' in '/'+p.filename and Path(p.filename).stem in stamps and p.filename.lower().endswith(('.png','.jpg','.jpeg'))]
            assert len(entries)==320,(camera,len(entries));images.extend(entries)
        total=sum(p.file_size for p in selected+images);assert total<=protocol['download_budget_bytes'],total
        remaining=sum(p.file_size for p in selected+images if not (out/p.filename).exists())
        assert shutil.disk_usage(out).free-remaining>=protocol['reserve_free_bytes']
        selected+=[] if metadata_only else images
        # Resolve early-file offsets against a bounded small range first.
        corrections=[];shifts={}
        # Adjacent members share an offset convention. This is a read-order
        # hint only: every filename and the complete member CRC still verify.
        selected.sort(key=lambda p:p.header_offset)
        manifest=[];begin=time.monotonic()
        for i,p in enumerate(selected,1):
            group=str(Path(p.filename).parent)
            correction=verified_local_offset(remote,p,shifts.get(group))
            shifts[group]=correction['verified_actual_offset']-correction['python_declared_offset']
            corrections.append(correction)
            rel=Path(p.filename);assert not rel.is_absolute() and '..' not in rel.parts
            destination=out/rel;destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                import zlib
                with destination.open('rb') as h:
                    crc=0
                    for b in iter(lambda:h.read(1048576),b''):crc=zlib.crc32(b,crc)
                assert crc & 0xffffffff==p.CRC,(p.filename,'existing CRC')
            else:
                content=z.read(p);assert len(content)==p.file_size
                temp=destination.with_suffix(destination.suffix+'.tmp');temp.write_bytes(content);temp.replace(destination)
            assert destination.stat().st_size==p.file_size
            manifest.append(dict(archive_member=p.filename,path=str(destination),bytes=p.file_size,SHA256=sha(destination),CRC32=p.CRC))
            if i%32==0:
                save(out/'PROGRESS.json',dict(status='DOWNLOADING_FIXED_SUBSET',done=i,total=len(selected),network_bytes=remote.bytes,seconds=time.monotonic()-begin));print('EXTERNAL_RANGE_FETCH',i,len(selected),remote.bytes,flush=True)
        save(out/'ZIP_OFFSET_COMPATIBILITY.json',dict(status='PASS',checks=corrections,CRC_required=True,no_annotation_or_image_content_modified=True))
        person_counts=[];id_occurrences={};sample=[]
        for p in annotations:
            js=read(out/p.filename);assert isinstance(js,list)
            person_counts.append(len(js))
            for a in js:
                assert 'personID' in a and 'views' in a
                key=str(a['personID']);id_occurrences[key]=id_occurrences.get(key,0)+1
            if len(sample)<2:sample.append(dict(frame=Path(p.filename).stem,annotation=js[:2]))
        result=dict(status='METADATA_READY' if metadata_only else 'DATA_READY',binding=binding(),protocol_SHA256=sha(REPORTS/'EXTERNAL_GENERALIZATION_PROTOCOL.json'),
                    archive_bytes=remote.size,network_bytes=remote.bytes,HTTP_range_requests=remote.requests,total_downloaded_members=len(manifest),
                    manifests=manifest,GT_metadata={'frames':320,'distinct_personID':len(id_occurrences),'IDs_repeated_across_frames':sum(n>1 for n in id_occurrences.values()),'person_count_min_max':[min(person_counts),max(person_counts)],'samples_for_schema_audit':sample},
                    tracking_results_viewed=False,GT_training_used=False,all_other_views_and_later_frames_unread=True)
        save(out/'RESULT_METADATA.json' if metadata_only else out/'RESULT.json',result)
        save(out/'PROGRESS.json',dict(status=result['status'],done=len(selected),total=len(selected)))
        print('EXTERNAL_WILDTRACK_READY',result['status'],len(manifest),result['GT_metadata'],flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--metadata-only',action='store_true');a=p.parse_args();main(a.metadata_only)

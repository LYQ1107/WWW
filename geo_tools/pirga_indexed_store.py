"""Bounded native feature/call storage and complete-epoch schedules.

No GT reader, tracker mutation or model dependency. The physical vector is stored
once; genuine Bank descriptor versions are separate immutable entries. Calls
retain every row/column in the original joint problem, using sparse nonzero
matrix entries, never GT-shortlisting. Offline supervision is a separate role.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import numpy as np

DIM=1152
INTERFACE='pirga-indexed-native/v1'
CALL_FIELDS={'scene','call_id','stage','query_keys','query_features','reference_features','reference_tracks',
             'candidate_ids','reference_block_sizes','matrix_shape','matrix_entries','thresholds','accepted_ids',
             'completed_image_ids'}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def sparse_matrix(value):
    a=np.asarray(value,dtype=np.float32)
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Nonfinite/incomplete native matrix')
    r,c=np.nonzero(a)
    return list(a.shape),[[int(i),int(j),float(a[i,j])] for i,j in zip(r,c)]


def dense_matrix(shape,entries):
    if len(shape)!=2 or any(type(n)!=int or n<0 for n in shape):raise ValueError('Matrix shape invalid')
    a=np.zeros(shape,dtype=np.float32);seen=set()
    for r,c,v in entries:
        if (r,c) in seen or not 0<=r<shape[0] or not 0<=c<shape[1] or not np.isfinite(v):raise ValueError('Invalid/duplicate matrix edge')
        seen.add((r,c));a[r,c]=v
    return a


class StoreWriter:
    def __init__(self,root,dtype='float16'):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.dtype=np.dtype(dtype)
        if self.dtype not in (np.dtype('float16'),np.dtype('float32')):raise ValueError('Unsupported feature precision')
        self.db=sqlite3.connect(self.root/'index.sqlite')
        self.db.execute('CREATE TABLE features (id INTEGER PRIMARY KEY, source TEXT UNIQUE, scene INTEGER, image INTEGER, detection INTEGER, frame INTEGER, camera INTEGER, variant TEXT, original_sha TEXT, original_dtype TEXT, stored_sha TEXT)')
        self.db.execute('CREATE TABLE calls (scene INTEGER, call_id INTEGER, body TEXT, PRIMARY KEY(scene,call_id))')
        self.db.execute('CREATE TABLE queries (source TEXT PRIMARY KEY, scene INTEGER, call_id INTEGER, row_id INTEGER)')
        self.bin=open(self.root/'native_vectors.bin','xb');self.n=0;self.calls=0;self.error_max=0.;self.error_sum=0.;self.elements=0

    def add_feature(self,scene,source,frame,camera,vector,variant=None):
        if type(scene)!=int or len(source)!=2 or any(type(v)!=int or v<0 for v in source) or type(frame)!=int or frame<1 or type(camera)!=int or camera not in (1,2):raise ValueError('Invalid physical provenance')
        x=np.asarray(vector)
        if x.shape!=(DIM,) or x.dtype not in (np.float32,np.float16) or not np.isfinite(x).all() or not np.linalg.norm(x.astype(np.float32)):raise ValueError('Full1152 finite native vector required')
        raw_sha=hashlib.sha256(x.dtype.str.encode()+x.tobytes()).hexdigest()
        key=json.dumps([scene,*source,variant],separators=(',',':'))
        row=self.db.execute('SELECT id, original_sha, frame, camera FROM features WHERE source=?',(key,)).fetchone()
        if row:
            if row[1:]!=(raw_sha,frame,camera):raise ValueError('Physical source overwritten; actual versions need distinct variant')
            return row[0]
        y=x.astype(self.dtype)
        if not np.isfinite(y).all():raise ValueError('Compression overflow')
        error=np.abs(y.astype(np.float32)-x.astype(np.float32));self.error_max=max(self.error_max,float(error.max()));self.error_sum+=float(error.sum(dtype=np.float64));self.elements+=DIM
        idx=self.n;self.n+=1;self.bin.write(y.tobytes())
        self.db.execute('INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?)',(idx,key,scene,*source,frame,camera,variant,raw_sha,str(x.dtype),hashlib.sha256(y.tobytes()).hexdigest()))
        return idx

    def add_call(self,c):
        validate_call(c,self.db)
        body=json.dumps(c,separators=(',',':'),allow_nan=False)
        self.db.execute('INSERT INTO calls VALUES (?,?,?)',(c['scene'],c['call_id'],body));self.calls+=1
        if c['stage']!='run_memory_tracker':
            for row,key in enumerate(c['query_keys']):self.db.execute('INSERT INTO queries VALUES (?,?,?,?)',(json.dumps(key,separators=(',',':')),c['scene'],c['call_id'],row))
        if self.calls%100==0:self.db.commit();self.bin.flush()

    def close(self,source_pins,expected_queries,status='ENGINEERING_ONLY'):
        self.bin.close();self.db.commit()
        count=self.db.execute('SELECT COUNT(*) FROM queries').fetchone()[0]
        if count!=expected_queries:raise ValueError('Incomplete query coverage')
        self.db.close()
        if status not in ('ENGINEERING_ONLY','SEALED_INPUTS_PENDING_SUPERVISION'):raise ValueError('No writer self-certifies full training admission')
        m=dict(interface=INTERFACE,status=status,native_feature_dim=DIM,stored_dtype=str(self.dtype),features=self.n,calls=self.calls,queries=count,
               quantization=dict(max_absolute_error=self.error_max,mean_absolute_error=self.error_sum/self.elements if self.elements else None,bitwise_native_precision=self.error_max==0),
               source_pins=source_pins,artifacts={n:sha(self.root/n) for n in ['index.sqlite','native_vectors.bin']},
               GT_in_input_store=False,supervision_role=None)
        (self.root/'manifest.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n');return m


def validate_call(c,db):
    if set(c)!=CALL_FIELDS:raise ValueError('Incomplete or labeled native call')
    if c['stage'] not in ('run_first_tracker_plus','run_global_tracker_plus','run_memory_tracker'):raise ValueError('Unknown solver stage')
    ids=c['candidate_ids'];q=c['query_keys'];refs=c['reference_features']
    if ids!=sorted(set(ids)) or len(c['thresholds'])!=len(ids) or len(c['query_features'])!=len(q) or len(c['accepted_ids'])!=len(q):raise ValueError('Joint solver layout changed')
    if len(refs)!=len(c['reference_tracks']) or sum(c['reference_block_sizes'])!=len(refs) or set(ids)!=set(c['reference_tracks']):raise ValueError('Missing full reference context')
    a=dense_matrix(c['matrix_shape'],c['matrix_entries'])
    if a.shape!=(len(q),len(ids)) or not np.isfinite(c['thresholds']).all():raise ValueError('Incomplete solver matrix/threshold')
    info={}
    for i in set(c['query_features']+refs):
        row=db.execute('SELECT scene,image,detection,frame,camera FROM features WHERE id=?',(i,)).fetchone()
        if row is None or row[0]!=c['scene']:raise ValueError('Missing/cross-scene native vector')
        info[i]=row
    for key,i in zip(q,c['query_features']):
        qo=info[i]
        if key!=[c['scene'],qo[1],qo[2]]:raise ValueError('Query physical key changed')
        for r in refs:
            ro=info[r]
            if not (ro[3]<qo[3] or ro[3]==qo[3] and ro[1] in c['completed_image_ids']):raise ValueError('Future/uncommitted history')
    from scipy.optimize import linear_sum_assignment
    rr,cc=linear_sum_assignment(-a);accepted=[-1]*len(q)
    for r,col in zip(rr,cc):
        if a[r,col]>c['thresholds'][col]:accepted[r]=ids[col]
    if accepted!=c['accepted_ids']:raise ValueError('Original native joint action changed')


class StoreReader:
    def __init__(self,root,manifest_sha):
        self.root=Path(root)
        if sha(self.root/'manifest.json')!=manifest_sha:raise ValueError('Unpinned store manifest')
        self.manifest=json.loads((self.root/'manifest.json').read_text());m=self.manifest
        if m['interface']!=INTERFACE or m['native_feature_dim']!=DIM or m['GT_in_input_store'] is not False:raise ValueError('Native input contract differs')
        for n,digest in m['artifacts'].items():
            p=(self.root/n).resolve()
            if not p.is_relative_to(self.root.resolve()) or sha(p)!=digest:raise ValueError('Input artifact corrupt')
        self.db=sqlite3.connect((self.root/'index.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        self.vectors=np.memmap(self.root/'native_vectors.bin',mode='r',dtype=m['stored_dtype'],shape=(m['features'],DIM)) if m['features'] else np.empty((0,DIM),np.float32)

    def feature(self,idx):return np.asarray(self.vectors[idx],dtype=np.float32).copy()

    def calls(self):
        for (body,) in self.db.execute('SELECT body FROM calls ORDER BY scene,call_id'):
            c=json.loads(body);validate_call(c,self.db);yield c


def epoch_order(eligible_by_scene,seed,epoch):
    """Every eligible physical query exactly once; scene-local shuffled streaming."""
    if type(epoch)!=int or epoch<1:raise ValueError('Epoch starts at1')
    rng=random.Random((seed<<16)+epoch);seen=set()
    scenes=sorted(eligible_by_scene);rng.shuffle(scenes)
    for scene in scenes:
        if scene==12 or scene not in (4,5,6,7,8,9,13,17,20,21,24):raise ValueError('Unauthorized fit scene')
        keys=[tuple(k) for k in eligible_by_scene[scene]]
        if any(k[0]!=scene for k in keys):raise ValueError('Cross-scene query')
        rng.shuffle(keys)
        for k in keys:
            if k in seen:raise ValueError('Duplicate physical training query')
            seen.add(k);yield k


def coverage(expected,visited):
    expected=list(map(tuple,expected));visited=list(map(tuple,visited))
    if len(set(expected))!=len(expected):raise ValueError('Eligibility duplicates')
    e=set(expected);v=set(visited)
    return dict(eligible_queries=len(e),unique_visited=len(v),visits=len(visited),missing=len(e-v),unexpected=len(v-e),
                duplicate_visits=len(visited)-len(v),coverage=len(e&v)/len(e) if e else None,
                full_epoch_pass=bool(e) and e==v and len(visited)==len(v),by_scene=dict(Counter(k[0] for k in visited)))

"""Bounded, read-only project storage/dependency inventory. Never deletes files."""
import argparse, collections, concurrent.futures, csv, datetime, hashlib, json, os
from pathlib import Path
import subprocess, time

BASE = Path('/data1/liuyeqiang/WWW_jev_phase7')
OUT = Path('/home/liuyeqiang/WWW_jev_phase8_runtime/20261008_v1')
FOUNDATION = Path('/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth')
B2 = Path('/home/liuyeqiang/WWW_jev_phase5_runtime/20261007/minimal_training/B2/calibration/model_calibrated.pth')
B2_SHA = 'f2aa3dd2b564d90bfbfb62dc0518b0b2107a931ed7134f8c0d19f5d152b94ed7'
CACHE = Path('/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train')
TRAIN = Path('/data/DATASETS/TRACKING/JDE/VisionTrack/train')
EXTENSIONS = {'.pth', '.pt', '.ckpt', '.safetensors'}
PRUNE = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'anaconda3'}
PERCEPTION_DIRS = set()

def perception_directory(path):
    records=path/'records'
    if not records.is_dir():return False
    # Cache directory names vary across historical runs. Identify the real
    # indexed video/frame/view tensor layout, rather than every .pt extension.
    names=[]
    for i,p in enumerate(records.iterdir()):
        names.append(p.name)
        if i>=7:break
    return bool(names)and all(n.startswith('video_')and'_frame_'in n and'_view_'in n and n.endswith('.pt')for n in names)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def command(args,timeout=120):
    try:
        p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
        return {'argv':args,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except subprocess.TimeoutExpired:
        return {'argv':args,'returncode':'TIMEOUT','stdout':'','stderr':'bounded command exceeded timeout'}

def scopes():
    trees=[]
    for block in subprocess.check_output(['git','worktree','list','--porcelain'],cwd=BASE,text=True).split('\n\n'):
        if block.startswith('worktree '): trees.append(Path(block.splitlines()[0][9:]))
    extra=[p for p in Path('/home/liuyeqiang').glob('WWW*') if p.is_dir()]
    extra += [Path('/home/liuyeqiang/checkpoint_archive_20261005'),Path('/home/liuyeqiang/old_proxy_trace_quarantine')]
    # Explicit project roots, no whole-disk discovery and no other users' homes.
    roots=sorted(set(trees+extra));roots=[p for p in roots if p.exists()]
    roots=[p for p in roots if not any(q!=p and q in p.parents for q in roots)]
    return trees,roots

def walk(root):
    device=root.stat().st_dev
    for directory,dirs,files in os.walk(root,followlinks=False):
        d=Path(directory)
        cached={n for n in dirs if perception_directory(d/n)}
        PERCEPTION_DIRS.update(d/n for n in cached)
        dirs[:]=[n for n in dirs if n not in PRUNE and n not in cached and not (d/n).is_symlink() and (d/n).stat().st_dev==device]
        for n in files:
            p=d/n
            if not p.is_symlink() and p.is_file(): yield p

def processes():
    rows=[];opened=collections.defaultdict(list);uncertain=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            if p.stat().st_uid!=os.getuid(): continue
            args=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
            if not args: continue
            cwd=os.readlink(p/'cwd')
            rows.append({'pid':int(p.name),'cwd':cwd,'argv':args[:1600]})
            for f in (p/'fd').iterdir():
                try:
                    target=os.readlink(f)
                    if target.startswith('/'): opened[target.removesuffix(' (deleted)')].append(int(p.name))
                except OSError: pass
        except (OSError,PermissionError): uncertain.append(int(p.name))
    return rows,opened,uncertain

def allocation(path):
    r=command(['du','-sx','--block-size=1',str(path)])
    return {'path':str(path),'allocated_bytes':int(r['stdout'].split()[0]) if r['returncode']==0 else None,'error':r['stderr']}

def reference_corpus(trees,roots):
    paths=set();skipped=[]
    for tree in trees:
        for sub in ('docs','configs','reproduction_tools','reports'):
            p=tree/sub
            if p.exists(): paths.update(walk(p))
    for root in roots:
        if 'runtime' in root.name:
            for p in walk(root):
                if p.suffix in {'.json','.md','.yaml','.yml','.sh','.py','.csv'}: paths.add(p)
    corpus=[];total=0
    for p in sorted(paths):
        if p.suffix not in {'.json','.md','.yaml','.yml','.sh','.py','.csv','.txt'}:continue
        if p.stat().st_size>2*1024*1024:
            skipped.append({'path':str(p),'reason':'large raw artifact; absence of reference not asserted'});continue
        if total+p.stat().st_size>384*1024*1024:
            skipped.append({'path':str(p),'reason':'bounded corpus limit; absence of reference not asserted'});continue
        try:
            data=p.read_text(errors='replace');total+=p.stat().st_size;corpus.append((str(p),data))
        except OSError: skipped.append({'path':str(p),'reason':'unreadable; retained dependencies are unknown'})
    return corpus,skipped,total

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report-dir',type=Path,default=OUT/'bootstrap/reports');args=ap.parse_args()
    dest=args.report_dir;dest.mkdir(parents=True,exist_ok=True)
    trees,roots=scopes();active,opened,uncertain=processes()
    commands=[command(['df','-hT']),command(['df','-i']),command(['du','-xhd1','/home/liuyeqiang']),command(['du','-xhd1','/data1/liuyeqiang'])]
    mounts=[]
    for path in ['/','/home/liuyeqiang','/data1/liuyeqiang','/data/DATASETS/TRACKING/JDE/VisionTrack/train']:
        s=os.statvfs(path);mounts.append({'path':path,'device':os.stat(path).st_dev,'total_bytes':s.f_blocks*s.f_frsize,'available_bytes':s.f_bavail*s.f_frsize,'used_bytes':(s.f_blocks-s.f_bfree)*s.f_frsize,'inodes':s.f_files,'free_inodes':s.f_favail})
    sizes=[allocation(p)for p in roots]+[allocation(CACHE),allocation(TRAIN),allocation(Path('/data1/liuyeqiang/WWW/.git/objects'))]
    corpus,skipped,corpus_bytes=reference_corpus(trees,roots)
    files=[];temporaries=[];all_checkpoints=[]
    for root in roots:
        for p in walk(root):
            if p.suffix in EXTENSIONS or (p.suffix in {'.bin','.state'} and any(x in p.name.lower() for x in ['optimizer','scheduler','ema'])):all_checkpoints.append(p)
            if p.suffix=='.tmp' or '.archive_staging' in p.parts:temporaries.append({'path':str(p),'size_bytes':p.stat().st_size,'classification':'D_UNKNOWN','delete_allowed':False,'reason':'no independently established redundant source/provenance'})
    hashes={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
        futures={pool.submit(sha,p):p for p in all_checkpoints}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            p=futures[f];hashes[str(p)]=f.result()
            if i%25==0:print(json.dumps({'phase':'HASH_INVENTORY','hashed':i,'total':len(futures)}),flush=True)
    hash_paths=collections.defaultdict(list)
    for p,h in hashes.items():hash_paths[h].append(p)
    for p in sorted(all_checkpoints):
        absolute=str(p);relative=[]
        for tree in trees:
            if tree in p.parents:relative.append(str(p.relative_to(tree)))
        exact=[];basename=[]
        for source,text in corpus:
            if absolute in text or any(x in text for x in relative):exact.append(source)
            elif p.name in text:basename.append(source)
        scope_protected= any(x in absolute for x in ['/WWW_jev_phase5/','/WWW_jev_phase6/','/WWW_jev_phase7/','/WWW_jev_phase5_runtime/','/WWW_jev_phase6_runtime/','/WWW_jev_phase7_runtime/']) or p==FOUNDATION or p.name=='model_20000.pth'
        if scope_protected:category='A_PROTECTED';reason='published Phase V/VI/VII reproducibility or explicit GMT backbone anchor'
        elif exact:category='B_REPRODUCIBILITY_REQUIRED';reason='explicit retained code/config/report reference; origin not inferred from inventory HEAD'
        else:category='D_UNKNOWN';reason='no proof of dispensability or complete training/report provenance; retain'
        code=[s for s in exact if any(x in s for x in ['/configs/','/reproduction_tools/'])]
        report=[s for s in exact if '/reports/'in s or '/docs/'in s]
        origin_head='UNKNOWN';inventory_head='UNKNOWN';run='UNKNOWN'
        for tree in trees:
            if tree in p.parents:
                inventory_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip();break
        for root in roots:
            if root in p.parents and 'runtime'in root.name:run=root.name+'/'+str(p.relative_to(root)).split('/')[0];break
        row={'path':absolute,'size_bytes':p.stat().st_size,'sha256':hashes[absolute],'mtime_utc':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),'run_id_scope':run,'source_commit':origin_head,'inventory_worktree_HEAD':inventory_head,'code_config_references':json.dumps(code),'report_manifest_references':json.dumps(report),'basename_only_references':json.dumps(basename[:20]),'active_process_ids':json.dumps(opened.get(absolute,[])),'reproduction_required':category in ['A_PROTECTED','B_REPRODUCIBILITY_REQUIRED'],'only_verified_content_copy':len(hash_paths[hashes[absolute]])==1,'verified_content_copies':json.dumps(hash_paths[hashes[absolute]]),'delete_breaks_audit':'YES'if category=='A_PROTECTED'else'POSSIBLE_NOT_DISPROVEN','classification':category,'delete_allowed':False,'classification_reason':reason}
        files.append(row)
    with(dest/'CHECKPOINT_INVENTORY.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(files[0]) if files else ['path']);w.writeheader();w.writerows(files)
    assert hashes[str(B2)]==B2_SHA
    save(dest/'CHECKPOINT_DEPENDENCY_AUDIT.json',{'status':'COMPLETE_CONSERVATIVE_RETAIN','scope_roots':[str(p)for p in roots],'checkpoint_count':len(files),'checkpoint_bytes':sum(r['size_bytes']for r in files),'classes':dict(collections.Counter(r['classification']for r in files)),'reference_corpus_files':len(corpus),'reference_corpus_bytes':corpus_bytes,'reference_scan_omissions':skipped,'absence_of_reference_claimed':False,'unknown_origin_commit_is_not_fabricated':True,'temporary_candidates':temporaries,'sha256_duplicate_groups':{h:p for h,p in hash_paths.items()if len(p)>1}})
    save(dest/'SAFE_DELETE_MANIFEST.json',{'status':'COMPLETE_NO_VERIFIED_ORPHANS','default_mode':'DRY_RUN','authorized_scope_roots':[str(p)for p in roots],'entries':[],'checkpoint_inventory_sha256':sha(dest/'CHECKPOINT_INVENTORY.csv'),'reason':'no file satisfies all ten orphan proofs; absence of reference is not proven for omitted raw artifacts; protected and unknown files remain','second_reference_and_process_check_required':True})
    (dest/'DELETE_DRY_RUN.md').write_text('# Safe cleanup dry run\n\nNo verified orphan checkpoint or temporary file satisfies every deletion condition. Delete list: empty. Planned release: 0 bytes. Protected, reproducible and unknown files are retained.\n')
    perceptions=[]
    for p in sorted(PERCEPTION_DIRS):
        row=allocation(p);index=p/'index.jsonl';row['role']='IMMUTABLE_PERCEPTION_TENSORS_NOT_MODEL_CHECKPOINTS';row['index_sha256']=sha(index)if index.is_file()else None;row['deleted']=False;perceptions.append(row)
    result={'status':'COMPLETE','captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'filesystems':mounts,'bounded_commands':commands,'project_directories':sizes,'checkpoint_bytes':sum(r['size_bytes']for r in files),'temporary_file_bytes':sum(r['size_bytes']for r in temporaries),'raw_archive_locations':[str(p)for p in roots if 'runtime'in p.name],'dataset_capacity_scope':'official TRAIN only; official TEST not inspected','perception_tensor_cache_inventory':perceptions,'checkpoint_extension_contract':'per-frame .pt files identified by existing perception-cache directory/index contract are data, not model checkpoints; cache paths are inventoried separately and protected','active_own_processes':active,'active_open_project_file_ids':{p:ids for p,ids in opened.items()if any(str(root)in p for root in roots)},'unreadable_own_processes':uncertain,'other_users_tasks_not_modified':True,'full_checkpoint_inventory_sha256':sha(dest/'CHECKPOINT_INVENTORY.csv'),'foundation_sha256':hashes.get(str(FOUNDATION)),'B2_sha256':hashes[str(B2)],'planned_experiment_output_root':str(OUT),'safe_delete_planned_bytes':0,'runtime_payload_lifecycle':'reuse perception; bounded forks; only best/last model snapshots; compact Git publication'}
    save(dest/'STORAGE_BEFORE.json',result);print(json.dumps({'status':'COMPLETE','checkpoints':len(files),'checkpoint_bytes':result['checkpoint_bytes'],'safe_delete_planned_bytes':0}),flush=True)

if __name__=='__main__':main()

"""Content-addressed temporary-ref transport; preserves all research history."""
import json,os,subprocess,time
from jev_phase7_common import *

REF='refs/heads/jev/phase7-evidence-transport-20261008'
BRANCH='refs/heads/jev/www-jev-phase7-causal-structured-20261008'
DEST=OUT/'git_transport';DEST.mkdir(exist_ok=True)
ENV=os.environ.copy()
SSH=subprocess.check_output(['git','config','--get','core.sshCommand'],cwd=ROOT,text=True).strip()
ENV['GIT_SSH_COMMAND']=SSH+' -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ConnectTimeout=20'

def command(args,**kwargs):return subprocess.check_output(['git',*args],cwd=ROOT,env=ENV,**kwargs)
def remote(ref):
    value=command(['ls-remote','origin',ref],text=True,timeout=50).strip()
    return value.split()[0]if value else None

def plan():
    parent=remote(BRANCH);assert parent
    head=command(['rev-parse','HEAD'],text=True).strip()
    listed=command(['rev-list','--objects',head,'^'+parent],text=True).splitlines()
    path_by_oid={line.split(' ',1)[0]:line.split(' ',1)[1]if' 'in line else ''for line in listed}
    checked=command(['cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input=('\n'.join(path_by_oid)+'\n').encode()).decode().splitlines()
    blobs=[]
    for line in checked:
        oid,kind,size=line.split();size=int(size)
        if kind=='blob'and size>=2*1024*1024:
            assert path_by_oid[oid].startswith('reports/JEV_PHASE7/'),(oid,path_by_oid[oid])
            assert size<95*1024*1024
            blobs.append({'oid':oid,'size':size,'source_path':path_by_oid[oid]})
    batches=[];current=[];total=0
    for blob in sorted(blobs,key=lambda r:(-r['size'],r['oid'])):
        if current and total+blob['size']>60*1024*1024:batches.append(current);current=[];total=0
        current.append(blob);total+=blob['size']
    if current:batches.append(current)
    entries=[];commits=[];previous=parent
    for index,batch in enumerate(batches):
        entries.extend(batch)
        listing=''.join(f"100644 blob {r['oid']}\t{r['oid']}.blob\n"for r in sorted(entries,key=lambda r:r['oid']))
        tree=command(['mktree'],input=listing.encode()).decode().strip()
        message=f'Phase VII verified evidence transport batch {index+1}; content-addressed copies, main history preserved\n'
        commit=command(['commit-tree',tree,'-p',previous],input=message.encode()).decode().strip()
        commits.append({'commit':commit,'blobs':batch,'bytes':sum(r['size']for r in batch)});previous=commit
    value={'status':'PLANNED','research_head':head,'original_remote_head':parent,'temporary_ref':REF,'batches':commits,
           'bytes':sum(r['size']for r in blobs),'blob_count':len(blobs),'research_history_rewritten':False}
    save(DEST/'plan.json',value);return value

if __name__=='__main__':
    protect();p=json.loads((DEST/'plan.json').read_text())if(DEST/'plan.json').exists()else plan()
    actual=remote(REF);all_commits=[r['commit']for r in p['batches']]
    if actual is not None:assert actual in all_commits,'foreign temporary ref; do not overwrite'
    start=all_commits.index(actual)+1 if actual else 0
    for index,batch in enumerate(p['batches'][start:],start):
        command(['update-ref',REF,batch['commit']])
        success=False
        for attempt in range(3):
            log=DEST/f'batch{index:03d}_attempt{attempt}.log'
            with log.open('w')as stream:
                try:code=subprocess.call(['git','push','--progress','origin',REF+':'+REF],cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
                except subprocess.TimeoutExpired:code=124
            current=remote(REF)
            if current==batch['commit']:success=True;break
        save(DEST/'progress.json',{'completed_batches':index+1 if success else index,'total_batches':len(all_commits),
            'last_commit':batch['commit'],'last_exit_code':code,'remote_verified':success})
        assert success,('failed evidence batch',index,code)
        print(json.dumps({'batch':index+1,'of':len(all_commits),'bytes':batch['bytes'],'remote_verified':True}),flush=True)
    p['status']='ALL_TRANSPORT_BLOBS_REMOTE_VERIFIED';save(DEST/'complete.json',p)
    print(json.dumps({'status':p['status'],'batches':len(all_commits),'bytes':p['bytes']}),flush=True)

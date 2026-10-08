"""Download bounded primary source files pinned before the literature cutoff."""
import ast,concurrent.futures,json,requests
from jev_phase7_common import *
PROXIES={'http':'http://127.0.0.1:17891','https':'http://127.0.0.1:17891'}
FILES={
 'zhongjiaru/CoopTrack':['projects/mmdet3d_plugin/cooptrack/modules/cross_agent_interaction.py','projects/mmdet3d_plugin/cooptrack/dense_heads/track_head_plugin/pf_runtime_tracker.py'],
 'FoxCanned/GMT':['gtr/modeling/meta_arch/gtr_rcnn.py','gtr/modeling/roi_heads/transformer.py'],
 '1941Zpf/TCEI':['TCEI/tcei.py','models/runtime_tracker.py'],
 'libingzheren/Jev-Mem':['memory/jev_mem_policies.py','memory/jev_mem_retrieval.py','memory/jev_client.py'],
 'ZimmyGao/openjev-rlcd':['openjev_rlcd/estimators.py','openjev_rlcd/train.py'],
 'AppliedMachineLearning-Lab/jev-benchmarking':['jev_benchmarking/metrics.py','jev_benchmarking/thresholds.py','jev_benchmarking/tasks/probes.py'],
 'MCG-NJU/MOTIP':['models/runtime_tracker.py','models/motip/id_decoder.py'],
 'yuxng/MDP_Tracking':['MDP_active.m','MDP_tracked.m','MDP_lost.m','MDP_initialize.m'],
 'kamkyu94/TrackTrack':[],
 'FoundationVision/ByteTrack':['yolox/tracker/byte_tracker.py','yolox/tracker/matching.py','yolox/tracker/basetrack.py','yolox/tracker/kalman_filter.py'],
}

def fetch(repo):
    folder=OUT/'external_sources'/repo.replace('/','__');folder.mkdir(parents=True,exist_ok=True)
    existing=folder/'tree.json'
    if existing.exists():meta=json.loads(existing.read_text());head=meta['commit'];date=meta['date'];tree=meta['tree']
    else:
        r=requests.get('https://api.github.com/repos/'+repo+'/commits',params={'until':'2026-10-07T23:59:59Z','per_page':1},proxies=PROXIES,timeout=30)
        if r.status_code!=200:return {'repo':repo,'status':'UNAVAILABLE','http_status':r.status_code}
        c=r.json()[0];head=c['sha'];date=c['commit']['committer']['date']
        tree=requests.get('https://api.github.com/repos/'+repo+'/git/trees/'+head,params={'recursive':1},proxies=PROXIES,timeout=30).json().get('tree',[])
        save(existing,{'repo':repo,'commit':head,'date':date,'tree':tree})
    chosen=FILES[repo]
    if repo=='FoundationVision/ByteTrack':head='d1bf0191adff59bc8fcfeaa0b33d3d1642552a99';date='2022-12-11T07:05:00Z'
    if not chosen:chosen=[x['path']for x in tree if x['type']=='blob' and x['path'].endswith('.py')and ('tracker.py'in x['path']or 'matching.py'in x['path']) and 'yolox' not in x['path'].lower()][:3]
    records=[]
    for file in chosen:
        url=f'https://raw.githubusercontent.com/{repo}/{head}/{file}'
        r=requests.get(url,proxies=PROXIES,timeout=30)
        item={'file':file,'url':url,'http_status':r.status_code}
        if r.status_code==200:
            p=folder/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
            item.update(sha256=sha(p),lines=len(r.text.splitlines()),local_path=str(p))
            if file.endswith('.py'):
                item['functions']=[n.name for n in ast.walk(ast.parse(r.text))if isinstance(n,ast.FunctionDef)]
        records.append(item)
    return {'repo':repo,'commit':head,'commit_date':date,'files':records,'status':'DOWNLOADED_NOT_YET_FUNCTION_AUDITED'}

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=5)as pool:rows=list(pool.map(fetch,FILES))
    save(REPORTS/'EXTERNAL_SOURCE_AUDIT.json',{'cutoff':'2026-10-07','repositories':rows})
    for r in rows:print(json.dumps({'repo':r['repo'],'commit':r.get('commit'),'status':r['status'],'files':[{k:v for k,v in f.items()if k in ('file','http_status','lines','functions')}for f in r.get('files',[])]}))

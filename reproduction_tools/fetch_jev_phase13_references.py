"""Pin specified public repositories; no imported/vendor code or remote instructions executed."""
import concurrent.futures,time
from jev_phase13_common import *
REPOS=['guanxuyu-sv/Visual-Jev','Liuziyu77/Valen','tinnel123666888/OmniJev','OmniJev/OneJev','TianyuCodings/NanoJev','arnodjiang/Vision-JEV','Xiaooolong/vev','IamBusy/OpenJev-Vision','sseanliu/Jev-Vision','mohit67890/imajev','TrackingLaboratory/CAMELTrack','MCG-NJU/MOTIP','MCG-NJU/MeMOTR','kamkyu94/TrackTrack','dvl-tum/SUSHI','NirAharon/BoT-SORT','FoundationVision/ByteTrack','GerardMaggiolino/Deep-OC-SORT','SysCV/qdtrack','noahcao/OC_SORT']
def fetch(repo):
    p=OUT/'external'/repo.replace('/','_');p.parent.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env['GIT_TERMINAL_PROMPT']='0'
    if not (p/'.git').exists():subprocess.run(['git','clone','--depth','1','--filter=blob:none','--no-checkout','https://github.com/'+repo+'.git',str(p)],env=env,check=True,timeout=150,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip();files=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'],cwd=p,text=True).splitlines()
    (p/'TREE.txt').write_text('\n'.join(files)+'\n')
    return {'repository':repo,'URL':'https://github.com/'+repo,'commit':commit,'checkout':str(p),'file_count':len(files),'license_paths':[f for f in files if Path(f).name.lower() in ['license','license.md','license.txt','copying']],'status':'SOURCE_PINNED_NOT_YET_AUDITED'}
def main():
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,r):r for r in REPOS}
        for future in concurrent.futures.as_completed(futures):
            try:r=future.result()
            except Exception as e:r={'repository':futures[future],'status':'UNAVAILABLE','reason':str(e)}
            results.append(r);save(OUT/'REFERENCE_DOWNLOAD_PROGRESS.json',{'status':'RUNNING','finished':len(results),'total':20,'results':results});print('REFERENCE',r['repository'],r['status'],flush=True)
    save(OUT/'REFERENCE_PINS.json',{'status':'COMPLETE_FETCH','repositories':sorted(results,key=lambda r:REPOS.index(r['repository']))})
if __name__=='__main__':main()

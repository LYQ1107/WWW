"""Unchanged official MATLAB CVIDF1/CVMA on arbitrary allowed complete scenes.

Only converter generalization removes the historical hardcoded three-DEV-scene
assertion. Sequential/interleaved conventions and official metric functions stay
byte-identical to the qualified Phase XIII/XIV evaluator.
"""
import argparse
import traceback
import numpy as np
from jev_oneweek_common import *


def prepare(manifest_path,output):
    manifest=read(manifest_path);jobs=[];evidence=[]
    assert len(manifest['sequences'])==2*len(manifest['scenes'])
    for scene in manifest['scenes']:
        sequences=sorted(s for s in manifest['sequences'] if s.startswith(scene+'_'))
        assert len(sequences)==2
        sources=[]
        for seq in sequences:
            gt=Path(manifest['trackeval_gt'])/seq/'gt/gt.txt'
            pred=Path(manifest['trackeval_trackers'])/'GMT/data'/(seq+'.txt')
            sources.append((np.loadtxt(gt,delimiter=',',ndmin=2),np.loadtxt(pred,delimiter=',',ndmin=2)))
            evidence.append(dict(sequence=seq,GT=ref(gt),prediction=ref(pred)))
        for convention in ['sequential','interleaved']:
            offset=0;gts=[];predictions=[]
            for index,(gt,pred) in enumerate(sources):
                length=int(max(gt[:,0].max(),pred[:,0].max() if len(pred) else 0))+1
                for source,target in [(gt,gts),(pred,predictions)]:
                    value=source[:,:6].copy()
                    value[:,0]=value[:,0]+offset if convention=='sequential' else value[:,0]*2+index
                    target.append(np.concatenate([value,-np.ones((len(value),4))],1))
                offset+=length
            folder=output/convention;folder.mkdir(parents=True,exist_ok=True);paths={}
            for kind,parts in [('gt',gts),('prediction',predictions)]:
                value=np.concatenate(parts);value=value[np.lexsort((value[:,1],value[:,0]))]
                path=folder/(scene+'_'+kind+'.txt')
                np.savetxt(path,value,delimiter=',',fmt=['%i','%i','%.2f','%.2f','%.2f','%.2f','%i','%i','%i','%i'])
                paths[kind]=path
            gt_directory=paths['gt'].with_suffix('');gt_directory.mkdir(exist_ok=True)
            length=max(int(np.concatenate(gts)[:,0].max()),int(np.concatenate(predictions)[:,0].max()))
            (gt_directory/'seqinfo.ini').write_text('[Sequence]\nname='+scene+'\nimDir=img1\nframeRate=30\nseqLength='+str(length)+'\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n')
            jobs.append(dict(scene=scene,convention=convention,groundtruth=str(paths['gt']),
                             prediction=str(paths['prediction']),gt_directory=str(gt_directory)))
    return jobs,evidence


def main(arm,seed,phase,videos):
    protect();assert set(videos)<=set(TRAIN+DEV)
    folder=OUT/'matlab_v1'/phase/arm/f'seed{seed}'/('_'.join(map(str,videos)))
    report_path=REPORTS/(f'MATLAB_{arm}_{phase}_SEED{seed}_VIDEOS_'+('_'.join(map(str,videos)))+'.json')
    if (folder/'RESULT.json').exists():
        completed=read(folder/'RESULT.json');assert completed['status']=='COMPLETE'
        for descriptor in completed['sources']+[completed['native']]:
            assert sha(descriptor['path'])==descriptor['SHA256']
        save(report_path,completed)
        print('REUSED_COMPLETED_EXACT_MATLAB_INPUT',arm,completed['CVIDF1'],completed['CVMA'],flush=True)
        return
    folder.mkdir(parents=True,exist_ok=False)
    sources=[];predictions=[]
    for video in videos:
        result=read(OUT/'online_v1'/phase/arm/f'seed{seed}'/f'video{video:02d}'/'RESULT.json')
        descriptor=result['raw_predictions'];assert sha(descriptor['path'])==descriptor['SHA256']
        predictions.extend(read(descriptor['path']));sources.append(ref(OUT/'online_v1'/phase/arm/f'seed{seed}'/f'video{video:02d}'/'RESULT.json'))
    if len(videos)==1:
        metric=read(sources[0]['path'])['metrics_source']['path']
        manifest=Path(metric).parent.parent/'prepared/manifest.json'
    else:
        from run_jev_phase10_closed_loop import metrics
        path=folder/'RAW_PREDICTIONS.json';save(path,predictions)
        score,evaluation=metrics(path,videos,folder/'pooled_TrackEval')
        manifest=evaluation.parent/'prepared/manifest.json'
    jobs,inputs=prepare(manifest,folder/'inputs')
    toolkit=Path('/home/liuyeqiang/WWW_jev_phase14/MOTChallengeEvalKit_cv_test/matlab_devkit')
    assert list(toolkit.rglob('*.mexa64'))
    previous=read(ROOT/'reports/JEV_PHASE14/OFFICIAL_MATLAB_RESULTS.json')['cases'][0]['result']
    prior=read(previous['path'])['official_toolkit_SHA256']
    for name,digest in prior.items():assert sha(toolkit/name)==digest,name
    matlab=Path('/data1/liuyeqiang/MATLAB/R2020a/bin/matlab')
    script=ROOT/'reproduction_tools/jev_phase13_official_matlab_eval.m'
    config=folder/'CONFIG.json';output=folder/'NATIVE.json';log=folder/'MATLAB.log'
    save(config,dict(kit=str(toolkit),jobs=jobs,output=str(output),benchmark='VisionTrack'))
    env=dict(os.environ,JEV_MATLAB_CONFIG=str(config),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    begin=time.monotonic()
    with log.open('w') as stream:
        subprocess.run([str(matlab),'-batch',"run('"+str(script)+"')"],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    native=read(output);assert native['status']=='COMPLETE'
    identities=[r['Identity'] for r in native['results'] if r['convention']=='sequential']
    clear=[r['CLEAR'] for r in native['results'] if r['convention']=='interleaved']
    counts={k:sum(v[k] for v in identities) for k in ['IDTP','IDFP','IDFN','n_gt','n_tr']}
    errors={k:sum(v[k] for v in clear) for k in ['fn','fp','id_switches','tp']}
    result=dict(status='COMPLETE',binding=binding('Unmodified official native MATLAB raw online multi-camera metrics'),
                arm=arm,seed=seed,phase=phase,videos=videos,sources=sources,
                CVIDF1=200*counts['IDTP']/(counts['n_gt']+counts['n_tr']),
                CVMA=100*(1-(errors['fn']+errors['fp']+errors['id_switches'])/counts['n_gt']),
                sequential_counts=counts,interleaved_CLEAR_counts=errors,native=ref(output),log=ref(log),
                converter_inputs=inputs,official_evaluator_hash_matches_frozen=True,
                prior_evaluator=previous,official_script=ref(script),native_version=native['version'],
                no_mean_of_video_metrics=True,seconds=time.monotonic()-begin)
    save(folder/'RESULT.json',result)
    save(report_path,result)
    print('OFFICIAL_MATLAB_COMPLETE',arm,videos,result['CVIDF1'],result['CVMA'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--phase',default='pilot');p.add_argument('--videos',type=int,nargs='+',required=True);a=p.parse_args()
    try:main(a.arm,a.seed,a.phase,a.videos)
    except Exception:
        failure('official_MATLAB',traceback.format_exc());raise

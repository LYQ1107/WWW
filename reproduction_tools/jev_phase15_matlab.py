"""Official unchanged MATLAB metrics on pooled raw Phase XV predictions."""
import argparse
import time
from jev_phase15_common import *
from evaluate_jev_phase13_official_matlab import prepare,MATLAB,KIT
from run_jev_phase10_closed_loop import metrics


def main(version=3):
    protect();out=OUT/f'pilot_pooled_live_v{version}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists()
    pooled=[];sources=[]
    for video in DEV:
        f=OUT/f'pilot_online_v{version}/F_full_seed20261009/live'/f'video{video:02d}/RESULT.json'
        r=read(f);assert r['status']=='COMPLETE' and r['live_images']
        p=r['raw_predictions'];assert sha(p['path'])==p['SHA256'];pooled.extend(read(p['path']));sources.append(ref(f))
    predictions=out/'RAW_PREDICTIONS.json';save(predictions,pooled)
    strict,path=metrics(predictions,DEV,out/'trackeval_raw')
    manifest=path/'tracking_eval_runtime_state/native/prepared/manifest.json'
    folder=out/'official_matlab_raw';jobs,inputs=prepare(manifest,folder)
    config=folder/'CONFIG.json';native=folder/'NATIVE.json';log=folder/'matlab.log'
    # The qualified compiled official toolkit is reused read-only; the previous
    # installation/source binaries are never rebuilt or changed by this task.
    toolkit=KIT
    if not list(toolkit.rglob('*.mexa64')):
        toolkit=Path('/home/liuyeqiang/WWW_jev_phase14/MOTChallengeEvalKit_cv_test/matlab_devkit')
    assert toolkit.exists() and list(toolkit.rglob('*.mexa64')),str(toolkit)
    save(config,dict(kit=str(toolkit),jobs=jobs,output=str(native),benchmark='VisionTrack'))
    env=os.environ.copy();env.update(JEV_MATLAB_CONFIG=str(config),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    script=ROOT/'reproduction_tools/jev_phase13_official_matlab_eval.m';start=time.monotonic()
    with log.open('w') as stream:
        subprocess.run([str(MATLAB),'-batch',"run('"+str(script)+"')"],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=read(native);assert result['status']=='COMPLETE'
    ids=[r['Identity'] for r in result['results'] if r['convention']=='sequential']
    clear=[r['CLEAR'] for r in result['results'] if r['convention']=='interleaved']
    counts={k:sum(v[k] for v in ids) for k in ['IDTP','IDFP','IDFN','n_gt','n_tr']}
    errors={k:sum(v[k] for v in clear) for k in ['fn','fp','id_switches','tp']}
    report=dict(status='COMPLETE',binding=binding(seed=20261009,dataset=sources,evaluator='unchanged TrackEval and native official MATLAB',
        scope='complete reused DEVELOPMENT, single diagnostic LAST pilot; no formal three-seed or independent-test claim'),
        version=version,checkpoint=read(sources[0]['path'])['checkpoint'],pooled_raw_TrackEval=strict,
        CVIDF1=200*counts['IDTP']/(counts['n_gt']+counts['n_tr']),CVMA=100*(1-(errors['fn']+errors['fp']+errors['id_switches'])/counts['n_gt']),
        sequential_identity_counts=counts,interleaved_CLEAR_counts=errors,unchanged_official_metrics=True,
        native_MATLAB=str(MATLAB),native_version=result['version'],native_report=ref(native),native_log=ref(log),
        toolkit_SHA256={str(p.relative_to(toolkit)):sha(p) for p in sorted(toolkit.rglob('*')) if p.suffix in ['.m','.cpp','.mexa64'] and p.is_file()},
        adapter=ref(script),input_manifest=ref(manifest),input_sources=inputs,raw_predictions=ref(predictions),
        raw_unfiltered=True,aggregation='sum per-scene counts; sequential CVIDF1/interleaved CVMA, no mean video score',seconds=time.monotonic()-start)
    save(out/'RESULT.json',report);save(REPORTS/'OFFICIAL_MATLAB_RESULTS.json',report)
    print('PHASE15_MATLAB_COMPLETE',version,strict,report['CVIDF1'],report['CVMA'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=3);a=p.parse_args();main(a.version)

"""Official unmodified native MATLAB metrics on frozen Phase XIV predictions."""
import argparse
import os
import time
from jev_phase14_common import *
from evaluate_jev_phase13_official_matlab import prepare,MATLAB,KIT


def ref(path):return dict(path=str(path),SHA256=sha(path))

def evaluate(variant,seed,phase):
    protect();source=binding();name=f'{variant}_seed{seed}';pooled=OUT/f'{phase}_pooled_v2'/name;out=OUT/f'{phase}_official_matlab_v2'/name;out.mkdir(parents=True,exist_ok=True)
    assert read(pooled/'RESULT.json')['status']=='COMPLETE'
    result=out/'RESULT.json'
    if result.exists():return read(result)
    begin=time.monotonic();reports={}
    for kind in ['raw_predictions','canonical_predictions']:
        manifest=pooled/kind/'tracking_eval_runtime_state/native/prepared/manifest.json';folder=out/kind
        jobs,inputs=prepare(manifest,folder);config=folder/'CONFIG.json';native=folder/'NATIVE.json';log=folder/'matlab.log'
        save(config,dict(kit=str(KIT),jobs=jobs,output=str(native),benchmark='VisionTrack'))
        env=os.environ.copy();env.update(JEV_MATLAB_CONFIG=str(config),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        script=ROOT/'reproduction_tools/jev_phase13_official_matlab_eval.m'
        with log.open('w') as h:subprocess.run([str(MATLAB),'-batch',"run('"+str(script)+"')"],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
        values=read(native);assert values['status']=='COMPLETE'
        identity=[r['Identity'] for r in values['results'] if r['convention']=='sequential'];clear=[r['CLEAR'] for r in values['results'] if r['convention']=='interleaved']
        counts={k:sum(v[k] for v in identity) for k in ['IDTP','IDFP','IDFN','n_gt','n_tr']};errors={k:sum(v[k] for v in clear) for k in ['fn','fp','id_switches','tp']}
        reports[kind]=dict(CVIDF1=200*counts['IDTP']/(counts['n_gt']+counts['n_tr']),CVMA=100*(1-(errors['fn']+errors['fp']+errors['id_switches'])/counts['n_gt']),
            Identity_counts=counts,interleaved_CLEAR_counts=errors,native_report=ref(native),native_log=ref(log),manifest=ref(manifest),input_sources=inputs,native=values)
    r=dict(status='COMPLETE',binding=source,variant=variant,seed=seed,phase=phase,reports=reports,native_MATLAB=str(MATLAB),native_version=values['version'],
        official_toolkit_SHA256={str(p.relative_to(KIT)):sha(p) for p in sorted(KIT.rglob('*')) if p.is_file() and p.suffix in ['.m','.cpp','.mexa64']},
        adapter_script=ref(ROOT/'reproduction_tools/jev_phase13_official_matlab_eval.m'),source_pooled_result=ref(pooled/'RESULT.json'),unchanged_official_metric_functions=True,
        aggregation='sum official per-scene counts; sequential CVIDF1 and interleaved CVMA; no mean-of-video score',
        input_policy='same historical official-converter formatting; max(GT,pred)+1 camera blocks; frame*n_views+index; XYWH two decimals; aux unknown -1; benchmarkVisionTrack skips MOT16-only class/visibility filtering',seconds=time.monotonic()-begin)
    save(result,r);print('PHASE14_NATIVE_MATLAB_COMPLETE',phase,variant,seed,reports['raw_predictions']['CVIDF1'],reports['raw_predictions']['CVMA'],flush=True);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--phase',default='formal');a=p.parse_args();evaluate(a.variant,a.seed,a.phase)

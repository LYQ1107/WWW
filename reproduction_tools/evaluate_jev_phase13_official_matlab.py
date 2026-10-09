"""Native official CVIDF1/CVMA on frozen predictions, separate from TrackEval.

Input formatting follows prepare_cross_view_eval.py: camera maximum over GT
and predictions plus one for sequential blocks, frame*n_views+view_index for
interleaved CVMA, XYWH rounded to two decimals and auxiliary columns=-1.
The original evaluator and repository MEX are untouched. VisionTrack does not
provide MOT16 class/visibility fields: benchmark='VisionTrack' skips MOT16-only
cleaning, which would otherwise discard true positives with sentinel vis=-1.
"""
from concurrent.futures import ThreadPoolExecutor
from jev_phase13_learning import *
import argparse
import time

MATLAB = Path('/data1/liuyeqiang/MATLAB/R2020a/bin/matlab')
KIT = ROOT / 'MOTChallengeEvalKit_cv_test/matlab_devkit'


def reference(path):
    return {'path': str(path), 'SHA256': sha(path)}


def rows(path):
    return np.loadtxt(path, delimiter=',', ndmin=2)


def prepare(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['sequences']) == 6 and len(manifest['scenes']) == 3
    assert sorted(manifest['seq_lengths'].values()) == sorted([1200,1200,1029,1029,1052,1052])
    jobs = []
    evidence = []
    for scene in manifest['scenes']:
        sequences = sorted(s for s in manifest['sequences'] if s.startswith(scene+'_'))
        sources = []
        for seq in sequences:
            gt = Path(manifest['trackeval_gt']) / seq / 'gt/gt.txt'
            pred = Path(manifest['trackeval_trackers']) / 'GMT/data' / (seq+'.txt')
            sources.append((rows(gt),rows(pred)))
            evidence.append({'sequence':seq,'groundtruth':reference(gt),'prediction':reference(pred)})
        for convention in ['sequential','interleaved']:
            offset = 0
            gts, predictions = [], []
            for index,(gt,pred) in enumerate(sources):
                length = int(max(gt[:,0].max(),pred[:,0].max() if len(pred) else 0))+1
                for source, target in [(gt,gts),(pred,predictions)]:
                    result = source[:,:6].copy()
                    result[:,0] = (result[:,0]+offset if convention=='sequential'
                                   else result[:,0]*len(sequences)+index)
                    target.append(np.concatenate([result,-np.ones((len(result),4))],axis=1))
                offset += length
            folder = output/convention
            folder.mkdir(parents=True,exist_ok=True)
            paths = {}
            for kind, parts in [('gt',gts),('prediction',predictions)]:
                value = np.concatenate(parts)
                value = value[np.lexsort((value[:,1],value[:,0]))]
                path = folder/(scene+'_'+kind+'.txt')
                np.savetxt(path,value,delimiter=',',fmt=['%i','%i','%.2f','%.2f','%.2f','%.2f','%i','%i','%i','%i'])
                paths[kind]=path
            # Official MOT16 preprocessing reads gt_directory+'.txt'.
            gt_directory = paths['gt'].with_suffix('')
            gt_directory.mkdir(exist_ok=True)
            length = max(int(np.concatenate(gts)[:,0].max()),int(np.concatenate(predictions)[:,0].max()))
            (gt_directory/'seqinfo.ini').write_text('[Sequence]\nname='+scene+'\nimDir=img1\nframeRate=30\nseqLength='+str(length)+'\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n')
            jobs.append({'scene':scene,'convention':convention,
                         'groundtruth':str(paths['gt']),'prediction':str(paths['prediction']),
                         'gt_directory':str(gt_directory)})
    return jobs,evidence


def evaluate_case(phase,name):
    original=name=='original_GMT'
    if original:phase='reference'
    source=(Path('/home/liuyeqiang/WWW_jev_phase12_runtime/20261009_v1/pooled_validation_v1/GMT_OFF')
            if original else OUT/f'{phase}_pooled_v1'/name)
    output=OUT/f'{phase}_official_matlab_v2'/name
    result=output/'RESULT.json'
    if result.exists():
        r=json.loads(result.read_text());assert r['status']=='COMPLETE';return r
    assert (source/'RESULT.json').exists() and MATLAB.exists()
    if original:
        frozen=json.loads((REPORTS/'GMT_BASELINE_FREEZE.json').read_text())
        prior_report=ROOT/'reports/JEV_PHASE12/MATCH_VALIDATION_RESULTS.json'
        assert sha(prior_report)==frozen['original_full_online_metrics_report_SHA256']
        assert json.loads((source/'RESULT.json').read_text())['pooled_metrics']==frozen['GMT_OFF_metrics']
    output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();reports={}
    for prediction_kind in ['raw_predictions','canonical_predictions']:
        source_kind=({'raw_predictions':'strict_online','canonical_predictions':'canonical_GMT_filtered'}[prediction_kind]
                     if original else prediction_kind)
        manifest=source/source_kind/'tracking_eval_runtime_state/native/prepared/manifest.json'
        folder=output/prediction_kind
        jobs,inputs=prepare(manifest,folder)
        config=folder/'CONFIG.json';native=folder/'NATIVE.json';log=folder/'matlab.log'
        save(config,{'kit':str(KIT),'jobs':jobs,'output':str(native),'benchmark':'VisionTrack'})
        env=os.environ.copy();env.update(JEV_MATLAB_CONFIG=str(config),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        script=ROOT/'reproduction_tools/jev_phase13_official_matlab_eval.m'
        with log.open('w') as handle:
            subprocess.run([str(MATLAB),'-batch',"run('"+str(script)+"')"],env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        values=json.loads(native.read_text());assert values['status']=='COMPLETE'
        identity=[r['Identity'] for r in values['results'] if r['convention']=='sequential']
        clear=[r['CLEAR'] for r in values['results'] if r['convention']=='interleaved']
        counts={k:sum(v[k] for v in identity) for k in ['IDTP','IDFP','IDFN','n_gt','n_tr']}
        errors={k:sum(v[k] for v in clear) for k in ['fn','fp','id_switches','tp']}
        reports[prediction_kind]={'CVIDF1':200*counts['IDTP']/(counts['n_gt']+counts['n_tr']),
                                  'CVMA':100*(1-(errors['fn']+errors['fp']+errors['id_switches'])/counts['n_gt']),
                                  'Identity_counts':counts,'interleaved_CLEAR_counts':errors,
                                  'native_report':reference(native),'native_log':reference(log),
                                  'manifest':reference(manifest),'input_sources':inputs,'native':values}
    r={'status':'COMPLETE','phase':phase,'case':name,'reports':reports,
       'binding':binding(),'native_MATLAB':str(MATLAB),'native_version':values['version'],
       'official_toolkit_SHA256':{str(p.relative_to(KIT)):sha(p) for p in sorted(KIT.rglob('*')) if p.is_file() and p.suffix in ['.m','.cpp','.mexa64']},
       'adapter_script':reference(ROOT/'reproduction_tools/jev_phase13_official_matlab_eval.m'),
       'benchmark_option':'VisionTrack; skip MOT16 class/visibility preprocessing because converter auxiliary fields are unknown -1, not measured MOT16 visibility',
       'source_pooled_result':reference(source/'RESULT.json'),
       'comparison_scope':('frozen original GMT Stage2 full system; separate perception/RPCE, not same-frontend causal ablation' if original else 'same Stage1 primary/onpolicy controller experiment'),
       'unchanged_official_metric_functions':True,
       'input_policy':'official converter sequential max(GT,pred)+1; interleaved frame*n_views+index; two-decimal XYWH; auxiliary -1; actual raw/canonical frozen predictions',
       'aggregation':'sum official per-scene counts; no mean-of-video scores',
       'wall_seconds':time.monotonic()-start,'heldout':'SEALED','official_TEST':False}
    save(result,r);print('PHASE13_NATIVE_MATLAB_CASE_COMPLETE',phase,name,r['reports']['raw_predictions']['CVIDF1'],r['reports']['raw_predictions']['CVMA'],flush=True)
    return r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true')
    parser.add_argument('--phase',choices=['formal','onpolicy'],default='formal')
    parser.add_argument('--case',default='cosine_seed20261009');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();protect()
    if not args.all:
        evaluate_case(args.phase,args.case);return
    cases=[('formal','cosine_seed20261009'),('reference','original_GMT')]
    cases += [('formal',f'{v}_seed{s}') for v in VARIANTS for s in SEEDS]
    cases += [('formal',f'full_seed{s}_no_calibration') for s in SEEDS]
    cases += [('onpolicy',f'{v}_seed{s}') for v in ['full','motip','camel','set_transformer'] for s in SEEDS
              if (OUT/f'onpolicy_pooled_v1/{v}_seed{s}/RESULT.json').exists()]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(evaluate_case,*case) for case in cases]
        results=[future.result() for future in futures]
    summaries={}
    for phase in ['formal','onpolicy','reference']:
        summaries[phase]={}
        for variant in ['cosine','original_GMT']+VARIANTS:
            selected=[r for r in results if r['phase']==phase and r['case'].split('_seed')[0]==variant and not r['case'].endswith('_no_calibration')]
            if selected:
                summaries[phase][variant]={key:{'mean':float(np.mean([r['reports']['raw_predictions'][key] for r in selected])),
                                                 'std_seed':float(np.std([r['reports']['raw_predictions'][key] for r in selected])),
                                                 'cases':[r['case'] for r in selected]}
                                           for key in ['CVIDF1','CVMA']}
    report={'status':'COMPLETE','binding':binding(),'cases':results,
            'strict_seed_summary':summaries,
            'metrics_scope':'actual native official MATLAB CVIDF1 and CVMA; raw primary, canonical separately',
            'native_smoke':reference(Path('/data1/liuyeqiang/matlab_R2020a_install_tools/official_metric_smoke.json')),
            'environment_audit':reference(REPORTS/'MATLAB_ENVIRONMENT_AUDIT.json'),
            'format_compatibility_audit':reference(REPORTS/'MATLAB_FORMAT_COMPATIBILITY_AUDIT.json'),
            'large_files_uploaded':False,'heldout':'SEALED','official_TEST':False}
    save(REPORTS/'OFFICIAL_MATLAB_CROSSVIEW.json',report)


if __name__=='__main__':
    main()

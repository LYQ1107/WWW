"""Run the original Python evaluator with an explicitly reported Octave transport."""
from pathlib import Path
import os,sys,argparse,json
r=Path('/data3/liuyeqiang/GMT_VisionTrack_repro');kit=r/'code/GMT/MOTChallengeEvalKit_cv_test'
p=argparse.ArgumentParser();p.add_argument('--gt-dir',required=True);p.add_argument('--res-dir',required=True);p.add_argument('--seq-file',required=True);p.add_argument('--output',required=True);args=p.parse_args()
output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
os.environ['GMT_REPRO_ROOT']=str(r);os.environ['GMT_OCTAVE_RESULTS_DIR']=str(output/'octave_intermediates');os.environ['GMT_OCTAVE_EXECUTABLE']=str(r/'tools/miniconda3/envs/GMT_eval/bin/octave');os.environ['OCTAVE_HOME']=str(r/'tools/miniconda3/envs/GMT_eval');os.environ['OMP_NUM_THREADS']='1'
os.chdir(kit);sys.path[:0]=[str(r/'tools/octave_bridge'),str(kit/'MOT'),str(kit)]
from evalMOT import MOT_evaluator
obj=MOT_evaluator()
overall,sequences=obj.run(benchmark_name='MOT16',gt_dir=str(Path(args.gt_dir).resolve()),res_dir=str(Path(args.res_dir).resolve()),seq_file=str(Path(args.seq_file).resolve()),save_pkl=str(output),eval_mode='test')
if obj.failed:raise RuntimeError('Official evaluator failed; inspect saved sequence logs')
obj.summary.to_csv(output/'summary.csv')
values={key:float(getattr(overall,key)) for key in overall.metrics}
(output/'metrics.json').write_text(json.dumps({'runtime':'GNU Octave transport, unchanged official metric sources','metrics':values,'sequences':[x.seqName for x in sequences]},indent=2))
print('OFFICIAL_CROSS_EVALUATOR_COMPLETE')

"""Evidence-based post-proof resource forecast; same frozen cache timing only."""
from gmt_jev_common import *
from gmt_jev_job import spent

def main():
 name='P5_mlp_learned_cached_full_GPU7'
 q=read(OUT/'learned_cache/QUEUE.json');p=pathlib.Path('/proc')/str(q['PID'])/'cmdline'
 try:signature=p.read_bytes()
 except FileNotFoundError:signature=b''
 assert b'gmt_jev_learned_cache_followup.py' not in signature,'Original owner must finish first'
 jobs=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs']
 assert name not in jobs,'Never duplicate or restart an existing timing case'
 assert q['status']=='SKIPPED_CACHE_TIMING_RESOURCE_PREFLIGHT'
 proof=read(REPORTS/'LEARNED_MLP_FEATURE_CACHE_PARITY.json')
 assert proof['status']=='PASS_EXACT_FULL_NATIVE_MLP_LEARNED' and proof['score_feature_pairs']==2067
 references=['P5_mlp_learned_full_GPU7','P5_rule_cached_full_image_GPU7']
 observed=[]
 for key in references:
  j=jobs[key];assert j['status']=='COMPLETE' and j['GPU']==7
  observed.append(dict(job=key,whole_job_seconds=j['seconds'],log=j['log'],source_HEAD=j['source']['HEAD']))
 # Full job wall times already include model load, all warmups, output hashing
 # and saving. Use the slower observed related workload, plus 10% and 10s.
 estimate=max(r['whole_job_seconds'] for r in observed)*1.10+10
 used=spent();remaining=(12-used)*3600
 status='APPROVED_WITH_UNCHANGED_HARD_CAP' if estimate<=remaining else 'SKIPPED_INSUFFICIENT_OBSERVED_RUNTIME_FORECAST'
 record=dict(status=status,previous_fixed_preflight_hours=.7,used_GPU_hours=used,remaining_GPU_seconds=remaining,
  estimated_seconds=estimate,observed_full_jobs=observed,formula='max(whole default MLP job, whole cached Rule job)*1.10 +10 seconds',
  no_cached_MLP_timing_observed_before_forecast=True,hard_GPU_hour_cap=12,
  model_checkpoint=proof['trained_checkpoint'],strict_parity=ref(REPORTS/'LEARNED_MLP_FEATURE_CACHE_PARITY.json'),
  original_queue_unchanged=q,scientific_success=False,
  scope='Scheduling estimate only; not a scientific hyperparameter change. Same fixed checkpoint, cache policy, video16, GPU7, three repeats and FPS convention. Actual executor terminates before the aggregate cap; partial timing cannot pass a gate.')
 save(REPORTS/'CACHE_TIMING_RESOURCE_REASSESSMENT.json',record)
 save(OUT/'cache_timing_tail/QUEUE.json',dict(PID=os.getpid(),status=status))
 if status!='APPROVED_WITH_UNCHANGED_HARD_CAP':return
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_job.py'),'--name',name,'--gpu','7',
  '--script','gmt_jev_speed.py','--','--kind','mlp_learned_cached','--gpu','7'],cwd=ROOT,check=True)
 save(OUT/'cache_timing_tail/QUEUE.json',dict(PID=os.getpid(),status='COMPLETE_FIXED_CACHED_MLP_TIMING'))
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_research_report.py')],cwd=ROOT,check=True)
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_publish.py'),'--message',
  'Record exact cached MLP complete timing under measured resource forecast'],cwd=ROOT,check=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  save(OUT/'cache_timing_tail/QUEUE.json',dict(PID=os.getpid(),status='FAILED_RETAINED_NO_AUTOMATIC_RESTART',reason=str(e)))
  raise

"""Finite native zero-residual prechecks; no training or policy search."""
from gmt_jev_common import *
from gmt_jev_job import spent

def main():
 for kind in ['mlp','attention','cic']:
  name=f'P0_learned_{kind.upper()}_zero_native_GPU7'
  jobs=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs']
  if name in jobs:
   assert jobs[name]['status']=='COMPLETE', 'Existing job must finish; never duplicate'
  else:
   reserve();assert spent()<12
   subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_job.py'),
    '--name',name,'--gpu','7','--script','gmt_jev_baseline.py','--',
    '--video','16','--frames','32','--zero','--zero-kind',kind,'--risk-mode','learned'],cwd=ROOT,check=True)
  assert read(REPORTS/f'LEARNED_GATE_ZERO_NATIVE_PARITY_{kind.upper()}.json')['status']=='PASS_EXACT'
 save(REPORTS/'LEARNED_GATE_ZERO_NATIVE_PARITY.json',dict(status='PASS_EXACT_ALL_THREE_HEADS_32_SCENE_PREFIX',
  cases=[ref(REPORTS/f'LEARNED_GATE_ZERO_NATIVE_PARITY_{k.upper()}.json') for k in ['mlp','attention','cic']],
  whole_video_new_learned_mode_parity=False,no_gradient_updates=True,independent_validation=False,source=binding()))
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_publish.py'),'--message','Verify all three learned gates at zero residual against native GMT states'],cwd=ROOT,check=True)

if __name__=='__main__':main()

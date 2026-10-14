"""Real child-process waiting, resource assignment and dependency failure flow."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid
from common import ROOT,RUNTIME


class Tests(unittest.TestCase):
    def test_wait_for_real_process_then_serial_resources_and_failed_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);marker=directory/'fixture_controller_wait_done'
            identifier='controller_fixture_'+uuid.uuid4().hex
            manifest=RUNTIME/'queues'/f'{identifier}_test'/'manifest.json'
            waiting_code=f'''import time,json
from pathlib import Path
path=Path({str(manifest)!r})
deadline=time.monotonic()+45
while time.monotonic()<deadline:
    if path.exists() and 'waits_for' in json.loads(path.read_text()):
        Path({str(marker)!r}).touch()
        break
    time.sleep(.05)
else:
    raise RuntimeError('controller did not observe the live fixture process')
'''
            waiting=subprocess.Popen([sys.executable,'-c',waiting_code])
            stages=[
                dict(id='cpu',cpu_only=True,command=['-c',f'import os;from pathlib import Path;assert Path({str(marker)!r}).exists();assert os.environ["CUDA_VISIBLE_DEVICES"]==""']),
                dict(id='fail',cpu_only=True,command=['-c','raise SystemExit(7)']),
                dict(id='skipped',cpu_only=True,depends_on=['fail'],command=['-c','raise RuntimeError("must not launch")']),
                dict(id='independent_gpu_env',command=['-c','import os;assert os.environ["CUDA_VISIBLE_DEVICES"]=="0"']),
            ]
            plan=directory/'plan.json';plan.write_text(json.dumps(dict(id=identifier,project_gpu_ids=['0'],queues=dict(test=stages),scope='CPU controller fixture; no GPU allocated and no image pipeline')))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            try:
                result=subprocess.run([sys.executable,str(ROOT/'geo_tools/run_research_queue.py'),'--plan',str(plan),'--queue','test','--gpu','0',
                    '--after-pid',str(waiting.pid),'--after-command-match','fixture_controller_wait_done'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)
                report=json.loads((RUNTIME/'queues'/f'{identifier}_test'/'manifest.json').read_text())
                self.assertEqual(report['waits_for']['pid'],waiting.pid)
                self.assertEqual(report['status'],'FINISHED_WITH_FAILURES')
                self.assertEqual([row['status'] for row in report['stages']],['PASS','FAILED','SKIPPED_FAILED_DEPENDENCY','PASS'])
                self.assertEqual(report['stages'][1]['exit_code'],7)
                self.assertNotIn('pid',report['stages'][2])
            finally:
                if waiting.poll() is None:waiting.terminate()
                waiting.wait(timeout=5)


if __name__=='__main__':unittest.main()

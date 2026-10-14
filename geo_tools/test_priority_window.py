import json,os,signal,subprocess,sys,tempfile,time,unittest,uuid
from pathlib import Path
from common import ROOT,RUNTIME


class Tests(unittest.TestCase):
    def test_current_child_finishes_before_priority_and_dispatcher_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);first=directory/'first_done';second=directory/'second_started';priority=directory/'priority_done'
            child_code=f'''import os,time
from pathlib import Path
path=Path('/proc')/str(os.getppid())/'stat'
deadline=time.monotonic()+45
while time.monotonic()<deadline:
    stat=path.read_text()
    if stat[stat.rfind(')')+2:].split()[0]=='T':
        time.sleep(.2);Path({str(first)!r}).touch();break
    time.sleep(.02)
else:raise RuntimeError('dispatcher was not held')
'''
            dispatcher_code=f'''import subprocess,sys,time
from pathlib import Path
subprocess.run([sys.executable,'-c',{child_code!r}],check=True)
Path({str(second)!r}).touch()
time.sleep(60)
'''
            dispatcher=subprocess.Popen([sys.executable,'-c',dispatcher_code,'--gpu','0','priority_dispatcher_fixture'],start_new_session=True)
            identifier='priority_fixture_'+uuid.uuid4().hex
            plan=directory/'plan.json';plan.write_text(json.dumps(dict(id=identifier,project_gpu_ids=['0'],queues=dict(gpu0=[dict(id='priority',cpu_only=True,
                command=['-c',f'from pathlib import Path;assert Path({str(first)!r}).exists();assert not Path({str(second)!r}).exists();Path({str(priority)!r}).touch()'])]),scope='CPU-only scheduling fixture; no GPU allocation')))
            try:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                result=subprocess.run([sys.executable,str(ROOT/'geo_tools/run_priority_window.py'),'--controller-pid',str(dispatcher.pid),'--controller-match','priority_dispatcher_fixture',
                    '--plan',str(plan),'--gpu','0','--tag',identifier],cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stderr)
                report=json.loads((RUNTIME/'priority_windows'/identifier/'manifest.json').read_text())
                self.assertTrue(report['original_dispatcher_resumed'])
                self.assertTrue(priority.exists())
                deadline=time.monotonic()+3
                while not second.exists() and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue(second.exists())
                queue=json.loads((RUNTIME/'queues'/f'{identifier}_gpu0'/'manifest.json').read_text())
                self.assertEqual(queue['status'],'FINISHED')
            finally:
                os.killpg(dispatcher.pid,signal.SIGTERM);dispatcher.wait(timeout=5)


if __name__=='__main__':unittest.main()

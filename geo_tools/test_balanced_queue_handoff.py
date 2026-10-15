import copy
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from run_balanced_queue_handoff import validate_plan


class Tests(unittest.TestCase):
    def fixture(self):
        stages=[dict(id='gate',command=['gate']),dict(id='current',command=['model']),
                dict(id='eval_current',command=['eval'],depends_on=['current']),
                dict(id='future',command=['model','--seed','16'],depends_on=['gate']),
                dict(id='eval_future',command=['eval_future'],depends_on=['future'])]
        plan=dict(original_queue='gpu0',existing_stage='current',queues={
            'gpu0':[{**stages[2],'depends_on':[]}],
            'gpu4':[{**stages[3],'depends_on':[]},stages[4]]})
        observed=[dict(id='gate',status='PASS'),dict(id='current',status='RUNNING')]
        return dict(queues={'gpu0':stages}),plan,observed

    def test_original_remaining_commands_are_accepted(self):
        validate_plan(*self.fixture())

    def test_duplicate_or_modified_stage_is_rejected(self):
        original,plan,observed=self.fixture()
        changed=copy.deepcopy(plan);changed['queues']['gpu4'].append(changed['queues']['gpu4'][0])
        with self.assertRaises(ValueError):validate_plan(original,changed,observed)
        changed=copy.deepcopy(plan);changed['queues']['gpu4'][0]['command'][-1]='15'
        with self.assertRaises(ValueError):validate_plan(original,changed,observed)

    def test_unverified_external_dependency_is_rejected(self):
        original,plan,observed=self.fixture();observed[0]['status']='FAILED'
        with self.assertRaises(ValueError):validate_plan(original,plan,observed)

    def test_stopped_parent_can_retire_after_child_without_launching_next(self):
        with tempfile.TemporaryDirectory() as folder:
            done=Path(folder)/'child_done';duplicate=Path(folder)/'next_launched'
            child_code=f"import time;from pathlib import Path;time.sleep(.4);Path({str(done)!r}).touch()"
            parent_code=f"import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);print(p.pid,flush=True);p.wait();Path({str(duplicate)!r}).touch()"
            parent=subprocess.Popen([sys.executable,'-u','-c',parent_code],stdout=subprocess.PIPE,text=True)
            try:
                int(parent.stdout.readline())
                os.kill(parent.pid,signal.SIGSTOP)
                deadline=time.monotonic()+3
                while not done.exists():
                    if time.monotonic()>deadline:self.fail('original child did not finish while parent held')
                    time.sleep(.01)
                os.kill(parent.pid,signal.SIGTERM);os.kill(parent.pid,signal.SIGCONT)
                self.assertEqual(parent.wait(timeout=3),-signal.SIGTERM)
                self.assertTrue(done.exists());self.assertFalse(duplicate.exists())
            finally:
                if parent.poll() is None:parent.kill();parent.wait()
                parent.stdout.close()


if __name__=='__main__':unittest.main()

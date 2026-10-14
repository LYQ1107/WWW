import unittest,time,numpy as np
from common import ROOT
from gtr.geometry.process_worker import ProcessGeometryWorker

def mock_factory(settings,counter):
    def compute(payload):
        with counter.get_lock():counter.value+=1
        time.sleep(settings.get('delay',.05))
        if payload.get('fail'):raise RuntimeError('simulated OOM')
        return dict(F=np.eye(3),valid=True,inlier_ratio=1.)
    return compute

class Tests(unittest.TestCase):
    def worker(self,**kwargs):
        worker=ProcessGeometryWorker(mock_factory,dict(delay=kwargs.pop('delay',.05)),gpu='',**kwargs)
        self.assertTrue(worker.wait_ready(timeout=15.));return worker

    def completion(self,worker,frame):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            result=worker.poll(frame)
            if result is not None or worker.pending is None:return result
            time.sleep(.01)
        self.fail('worker completion deadline')

    def test_one_pending_and_completion_boundary(self):
        worker=self.worker(delay=.15)
        try:
            self.assertTrue(worker.submit(0,{}));self.assertFalse(worker.submit(1,{}))
            self.assertIsNone(worker.poll(0));result=self.completion(worker,1)
            self.assertEqual(result['source_frame'],0)
            self.assertLessEqual(result['completed_at'],time.monotonic())
            self.assertEqual(worker.encoder_calls,1)
        finally:worker.close()

    def test_failure_and_future_or_stale_neutral(self):
        worker=self.worker(max_age=2)
        try:
            worker.submit(0,dict(fail=True));self.assertIsNone(self.completion(worker,1));self.assertEqual(worker.stats['failed'],1)
            worker.submit(2,{});self.assertIsNone(self.completion(worker,1));self.assertEqual(worker.stats['future_rejected'],1)
            worker.submit(3,{});self.assertIsNone(self.completion(worker,6));self.assertEqual(worker.stats['expired'],1)
        finally:worker.close()

    def test_hung_task_timeout_and_bounded_shutdown(self):
        worker=self.worker(delay=30.,timeout=.05,shutdown_timeout=.2)
        try:
            worker.submit(0,{});time.sleep(.08)
            self.assertIsNone(worker.poll(1));self.assertEqual(worker.stats['timeouts'],1)
            self.assertFalse(worker.process.is_alive());self.assertFalse(worker.submit(2,{}))
        finally:
            started=time.monotonic();worker.close();self.assertLess(time.monotonic()-started,1.)

if __name__=='__main__':unittest.main()

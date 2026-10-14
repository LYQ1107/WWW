"""One pending/current task. Finished results never backfill earlier decisions."""
from concurrent.futures import ThreadPoolExecutor
import time
from .geometry_state import GeometryState

class GeometryWorker:
    def __init__(self,compute,max_age=20,timeout=5.):
        self.compute=compute;self.max_age=max_age;self.timeout=timeout
        self.pool=ThreadPoolExecutor(max_workers=1);self.future=None;self.submitted=None;self.source_frame=None
        self.stats=dict(submitted=0,dropped_busy=0,expired=0,failed=0)

    def submit(self,frame,payload):
        if self.future is not None:self.stats['dropped_busy']+=1;return False
        self.submitted=time.monotonic();self.source_frame=frame
        def work():
            result=self.compute(payload)
            return GeometryState(frame,time.monotonic(),result.get('F'),result.get('valid',False),result.get('confidence',0.))
        self.future=self.pool.submit(work);self.stats['submitted']+=1;return True

    def poll(self,current_frame):
        if self.future is None or not self.future.done():return None
        now=time.monotonic();future=self.future;self.future=None
        try:state=future.result()
        except Exception:self.stats['failed']+=1;return None
        if now-self.submitted>self.timeout or not state.usable(current_frame,now,self.max_age):self.stats['expired']+=1;return None
        return state

    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)

"""Spawned geometry worker: one in-flight payload, bounded timeout/shutdown.

CPU-only payload/result IPC; CUDA contexts and RNG streams stay independent.
"""
import multiprocessing as mp,os,time,traceback

def _process_main(connection,factory,settings,counter):
    try:
        compute=factory(settings,counter)
        connection.send(dict(kind='ready',at=time.monotonic(),peak_gpu_bytes=getattr(compute,'peak_gpu_bytes',None)))
        while True:
            message=connection.recv()
            if message is None:break
            frame,payload=message
            try:
                result=compute(payload)
                response=dict(kind='result',source_frame=frame,completed_at=time.monotonic(),result=result,error=None)
            except Exception as exc:
                response=dict(kind='result',source_frame=frame,completed_at=time.monotonic(),result=None,error=f'{type(exc).__name__}: {exc}')
            connection.send(response)
    except (EOFError,BrokenPipeError):pass
    except Exception as exc:
        try:connection.send(dict(kind='startup_error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()[-1500:]))
        except (EOFError,BrokenPipeError,OSError):pass
    finally:connection.close()

class ProcessGeometryWorker:
    def __init__(self,factory,settings,gpu,max_age=20,timeout=5.,shutdown_timeout=1.,startup_timeout=60.):
        self.max_age=max_age;self.timeout=timeout;self.shutdown_timeout=shutdown_timeout
        context=mp.get_context('spawn');self.connection,child=context.Pipe();self.counter=context.Value('i',0)
        self.process=context.Process(target=_process_main,args=(child,factory,settings,self.counter),daemon=True)
        self.pending=None;self.ready=False;self.disabled=False;self.trace=[];self.error=None
        self.ready_peak_gpu_bytes=None
        self.started_at=time.monotonic();self.startup_timeout=startup_timeout
        self.stats=dict(submitted=0,dropped_busy=0,dropped_not_ready=0,received=0,expired=0,failed=0,timeouts=0,future_rejected=0)
        # The child must see its GPU before importing gtr/torch during spawn.
        # The parent's existing CUDA context is unchanged by this env snapshot.
        previous=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
        try:self.process.start()
        finally:
            if previous is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=previous
        child.close()

    @property
    def encoder_calls(self):return self.counter.value

    def poll_ready(self):
        if self.ready:return True
        if self.disabled:return False
        if self.connection.poll():
            try:message=self.connection.recv()
            except (EOFError,OSError):message=dict(kind='startup_error',error='worker terminated during startup')
            if message['kind']=='ready':
                self.ready=True;self.ready_peak_gpu_bytes=message.get('peak_gpu_bytes');self.trace.append(dict(event='ready',at=message['at']));return True
            self.error=message.get('error');self.stats['failed']+=1;self._terminate()
        elif not self.process.is_alive() or time.monotonic()-self.started_at>self.startup_timeout:
            self.error='worker startup failed or expired';self.stats['timeouts']+=1;self._terminate()
        return False

    def wait_ready(self,timeout=60.):
        if self.ready:return True
        if self.connection.poll(timeout):
            try:message=self.connection.recv()
            except (EOFError,OSError):message=dict(kind='startup_error',error='worker terminated during startup')
            if message['kind']=='ready':self.ready=True;self.ready_peak_gpu_bytes=message.get('peak_gpu_bytes');return True
            self.error=message.get('error');self.stats['failed']+=1
        else:self.error='worker startup timeout';self.stats['timeouts']+=1
        self._terminate();return False

    def submit(self,frame,payload):
        if self.disabled or not self.poll_ready():self.stats['dropped_not_ready']+=1;return False
        if self.pending is not None:self.stats['dropped_busy']+=1;return False
        if not self.process.is_alive():
            self.stats['failed']+=1;self.error='worker exited before submission';self._terminate();return False
        submitted=time.monotonic()
        try:self.connection.send((frame,payload))
        except (OSError,BrokenPipeError):
            self.stats['failed']+=1;self.error='worker IPC submission failed';self._terminate();return False
        self.pending=(frame,submitted);self.stats['submitted']+=1
        self.trace.append(dict(event='submitted',frame=frame,at=self.pending[1]));return True

    def poll(self,current_frame,now=None):
        self.poll_ready()
        if self.pending is None or self.disabled:return None
        realtime=now is None
        now=time.monotonic() if now is None else now
        if not self.connection.poll():
            if now-self.pending[1]>self.timeout:
                self.stats['timeouts']+=1;self.error='in-flight worker timeout';self._terminate()
            elif not self.process.is_alive():
                self.stats['failed']+=1;self.error='worker exited with pending task';self._terminate()
            return None
        source,submitted=self.pending;self.pending=None
        try:message=self.connection.recv()
        except (EOFError,OSError):
            self.stats['failed']+=1;self._terminate();return None
        if realtime:now=time.monotonic()
        self.stats['received']+=1
        self.trace.append(dict(event='received',source_frame=source,current_frame=current_frame,completed_at=message.get('completed_at'),observed_at=now))
        if message.get('error'):
            self.stats['failed']+=1;self.error=message['error'];return None
        if message.get('source_frame')!=source or source>current_frame or message['completed_at']>now:
            self.stats['future_rejected']+=1;return None
        if current_frame-source>self.max_age or now-submitted>self.timeout:
            self.stats['expired']+=1;return None
        return message

    def _terminate(self):
        self.disabled=True;self.pending=None
        if self.process.is_alive():self.process.terminate()
        self.process.join(timeout=self.shutdown_timeout)
        if self.process.is_alive():self.process.kill();self.process.join(timeout=self.shutdown_timeout)

    def close(self):
        start=time.monotonic()
        if self.process.is_alive():
            try:self.connection.send(None)
            except (EOFError,OSError,BrokenPipeError):pass
            self.process.join(timeout=self.shutdown_timeout)
        if self.process.is_alive():self._terminate()
        else:self.process.join(timeout=self.shutdown_timeout)
        self.connection.close();self.disabled=True
        return time.monotonic()-start

def vggt_camera_factory(settings,counter):
    # Imports intentionally occur in the spawned child only.
    import sys,torch,cv2,numpy as np
    sys.path.insert(0,settings['vggt_source']);cv2.setNumThreads(4)
    from .vggt_adapter import FrozenVGGT
    from .roi_feature_extractor import PadMapping
    from .geometry_confidence import estimate_background
    from .camera_geometry import validate_camera_fundamental
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    adapter=FrozenVGGT(settings['checkpoint'],device='cuda',layers=[17])
    def compute(payload):
        records=payload['records'];boxes=[torch.tensor(b,dtype=torch.float32).reshape(-1,4) for b in payload['boxes']]
        mappings=[PadMapping(r['width'],r['height']) for r in records];mapped=[m.boxes(b) for m,b in zip(mappings,boxes)]
        images=load_and_preprocess_images([r['file_name'] for r in records],mode='pad')
        for v,b in enumerate(mapped):
            for x1,y1,x2,y2 in b.round().int().tolist():images[v,:,max(0,y1):min(518,y2),max(0,x1):min(518,x2)]=1.
        with counter.get_lock():counter.value+=1
        _,_,meta=adapter.extract(images,mapped,modes=(),camera=True)
        E,K=pose_encoding_to_extri_intri(meta['pose'],(518,518))
        gray=[cv2.imread(r['file_name'],cv2.IMREAD_GRAYSCALE) for r in records]
        background=estimate_background(*gray,*[b.numpy() for b in boxes],method='sift',mask_targets=True)
        result=validate_camera_fundamental(E.numpy()[0],K.numpy()[0],mappings,background)
        result.update(camera_order=tuple(r['view_id'] for r in records),peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved())
        return result
    compute.peak_gpu_bytes=torch.cuda.max_memory_allocated()
    return compute

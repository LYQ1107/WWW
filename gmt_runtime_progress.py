"""CPU-only progress records and on-demand stack dumps; no tensor/gradient changes."""
from pathlib import Path
import atexit,faulthandler,json,os,signal,time
_files=[]

def install(model,output,rank):
    directory=Path(output)/'runtime';directory.mkdir(parents=True,exist_ok=True)
    progress=(directory/f'progress_rank{rank}.jsonl').open('a',buffering=1)
    stacks=(directory/f'stacks_rank{rank}.log').open('a',buffering=1)
    _files.extend([progress,stacks])
    faulthandler.register(signal.SIGUSR1,file=stacks,all_threads=True)
    atexit.register(faulthandler.cancel_dump_traceback_later)
    def record(event):
        from detectron2.utils.events import get_event_storage
        storage=get_event_storage()
        progress.write(json.dumps({'pid':os.getpid(),'rank':rank,'iteration':storage.iter,'event':event,'time':time.time()})+'\n')
    def before(module,args):
        record('forward_start')
        faulthandler.dump_traceback_later(300,repeat=True,file=stacks)
    def after(module,args,result):
        record('forward_done')
    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)

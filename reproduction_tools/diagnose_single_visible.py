"""Ten global ranks, one visible physical GPU per spawned worker; diagnostic only."""
from pathlib import Path
import datetime
import json
import os
import re
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(rank, args, seeds, parent, gpu_uuids, dist_url):
    # Set visibility before importing training code or querying the CUDA runtime.
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuids[rank]
    import diagnose_native_backward as observation
    import torch.distributed as dist
    from detectron2.utils import comm
    torch = observation.torch
    assert not torch.cuda.is_initialized(), 'CUDA context initialized before isolation'
    assert torch.cuda.device_count() == 1, 'Expected exactly one visible CUDA device'
    dist.init_process_group(backend='gloo', init_method=dist_url, world_size=10,
                            rank=rank, timeout=datetime.timedelta(minutes=30))
    comm.create_local_process_group(1)
    torch.cuda.set_device(0)
    comm.synchronize()
    directory = observation.OUT/'runtime'
    directory.mkdir(parents=True, exist_ok=True)
    (directory/f'visibility_rank{rank}.json').write_text(json.dumps({
        'pid': os.getpid(), 'global_rank': comm.get_rank(), 'world_size': comm.get_world_size(),
        'local_rank': comm.get_local_rank(), 'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'logical_cuda_device': torch.cuda.current_device(), 'device_count': torch.cuda.device_count()}))
    observation.worker(args, seeds, parent)
    dist.destroy_process_group()


if __name__ == '__main__':
    if os.environ.get('GMT_POST_BACKWARD_CPU') != '1':
        raise RuntimeError('This candidate requires the validated post-backward averaging path')
    import diagnose_native_backward as observation
    import torch.multiprocessing as mp
    torch = observation.torch
    if observation.OUT.exists():raise RuntimeError('Refusing existing diagnostic output')
    listing = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping = {int(line.split(',')[0]):line.split(',')[1].strip() for line in listing.splitlines()}
    gpu_uuids = [mapping[i] for i in range(10)]
    assert len(set(gpu_uuids)) == 10
    seeds = []
    for rank in range(10):
        name='log.txt'+(f'.rank{rank}' if rank else '')
        seeds.append(int(re.search(r'Using a generated random seed (\d+)',(ROOT/'outputs/stage1'/name).read_text()).group(1)))
    args=observation.default_argument_parser().parse_args(['--num-gpus','10','--config-file',
        str(observation.REPO/'configs/VISION_stage1.yaml'),'SOLVER.IMS_PER_BATCH','10',
        'SOLVER.TRAIN_ITER','100','MODEL.WEIGHTS',str(ROOT/'checkpoints/backbone/CH_FPN_1x_key_adapted.pth'),
        'OUTPUT_DIR',str(observation.OUT)])
    dist_url='tcp://127.0.0.1:{}'.format(torch.randint(12000,59000,(1,)).item())
    (ROOT/'manifests'/f'single_visible_{observation.VARIANT}_launch.json').write_text(json.dumps({
        'pid':os.getpid(),'formal_training':False,'global_world_size':10,'global_batch':10,
        'gpu_uuids':gpu_uuids,'recorded_rank_seeds':seeds,'note':'Local rank becomes0 on every process; each physical GPU remains distinct. Model and global sampler/gradient averaging unchanged.'},indent=2))
    done=threading.Event();watcher=threading.Thread(target=observation.monitor,args=(done,),daemon=True);watcher.start()
    try:
        mp.spawn(bootstrap,args=(args,seeds,os.getpid(),gpu_uuids,dist_url),nprocs=10,join=True)
        print('DIAGNOSTIC_100_UPDATES_COMPLETED_NOT_FORMAL',flush=True)
    finally:
        done.set();watcher.join(timeout=50)

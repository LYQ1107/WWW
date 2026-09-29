"""Select a supported process-group backend for shared-GPU execution."""
import os
import torch.distributed as dist


def install_launch_backend():
    requested = os.environ.get('GMT_DISTRIBUTED_BACKEND')
    if os.environ.get('GMT_CPU_DDP_INIT') == '1':
        if requested != 'gloo' or os.environ.get('GMT_CPU_COLLECTIVES') != '1':
            raise RuntimeError('CPU DDP initialization requires CPU-staged Gloo collectives')
        install_cpu_ddp_initialization()
    if os.environ.get('GMT_CPU_COLLECTIVES') == '1':
        if requested != 'gloo' or os.environ.get('GMT_SYNC_DDP_BUCKETS') == '1':
            raise RuntimeError('CPU collectives require Gloo and no other DDP hook')
        install_cpu_collectives()
    if not requested:
        return
    if requested == 'nccl':
        return  # Keep Detectron2's original default backend, without monkey-patching.
    if requested != 'gloo':
        raise ValueError('Only gloo override or native nccl is supported')
    if getattr(dist.init_process_group, '_gmt_backend_override', False):
        return
    original = dist.init_process_group
    def initialize(*args, **kwargs):
        # Detectron2 passes its default NCCL backend as a keyword.
        if args:
            raise RuntimeError('Unexpected positional backend; refusing ambiguous override')
        if str(kwargs.get('backend', '')).lower() == 'nccl':
            kwargs['backend'] = requested
        return original(**kwargs)
    initialize._gmt_backend_override = True
    dist.init_process_group = initialize


def install_synchronous_bucket_hook(model):
    """Use PyTorch's standard averaging hook, with a host/device fence per bucket."""
    import torch
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
    def synchronized_average(process_group, bucket):
        value = allreduce_hook(process_group, bucket).wait()
        if bucket.buffer().is_cuda:
            torch.cuda.synchronize(bucket.buffer().device)
        completed = torch.futures.Future()
        completed.set_result(value)
        return completed
    model.register_comm_hook(dist.group.WORLD, synchronized_average)


def install_cpu_collectives():
    """Stage synchronous CUDA scalar/logging reductions through Gloo CPU tensors."""
    if getattr(dist.all_reduce, '_gmt_cpu_staged', False):
        return
    original_all_reduce, original_reduce = dist.all_reduce, dist.reduce

    def stage(tensor, operation, **kwargs):
        if not tensor.is_cuda:
            return operation(tensor, **kwargs)
        if kwargs.get('async_op') or tensor.requires_grad:
            raise RuntimeError('CPU staging only supports synchronous non-autograd collectives')
        if dist.get_backend(kwargs.get('group')) != 'gloo':
            raise RuntimeError('CPU staging requires a Gloo process group')
        host = tensor.cpu()
        result = operation(host, **kwargs)
        tensor.copy_(host)
        return result

    def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        return stage(tensor, original_all_reduce, op=op, group=group, async_op=async_op)

    def reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
        return stage(tensor, original_reduce, dst=dst, op=op, group=group, async_op=async_op)

    all_reduce._gmt_cpu_staged = True
    dist.all_reduce, dist.reduce = all_reduce, reduce


def install_cpu_bucket_hook(model):
    """Average each DDP gradient bucket on CPU, retaining CUDA model computation."""
    import torch
    def average(process_group, bucket):
        if dist.get_backend(process_group) != 'gloo':
            raise RuntimeError('CPU gradient averaging requires Gloo')
        value = bucket.buffer()
        # clone also keeps the CPU-only equivalence test from aliasing the bucket.
        host = value.detach().to(device='cpu', copy=True)
        host.div_(dist.get_world_size(process_group))
        dist.all_reduce(host, group=process_group)
        value.copy_(host)
        completed = torch.futures.Future()
        completed.set_result(value)
        return completed
    model.register_comm_hook(dist.group.WORLD, average)


def install_cpu_ddp_initialization():
    """Stage DDP shape validation and state broadcasts through Gloo on CPU."""
    import torch
    if getattr(dist._broadcast_coalesced, '_gmt_cpu_staged', False):
        return
    original_verify = dist._verify_params_across_processes
    original_broadcast = dist._broadcast_coalesced

    def host_copies(process_group, tensors):
        if dist.get_backend(process_group) != 'gloo':
            raise RuntimeError('CPU DDP initialization requires Gloo')
        return [tensor.detach().to(device='cpu', copy=True) for tensor in tensors]

    def verify(process_group, tensors, logger=None):
        return original_verify(process_group, host_copies(process_group, tensors), logger)

    def broadcast(process_group, tensors, buffer_size, src=0):
        host = host_copies(process_group, tensors)
        result = original_broadcast(process_group, host, buffer_size, src)
        with torch.no_grad():
            for tensor, value in zip(tensors, host):
                tensor.copy_(value)
        return result

    broadcast._gmt_cpu_staged = True
    dist._verify_params_across_processes = verify
    dist._broadcast_coalesced = broadcast

"""Candidate CPU gradient synchronization after backward; no DDP CUDA collectives."""
import torch
import torch.distributed as dist


@torch.no_grad()
def synchronize_model(model):
    if dist.get_backend() != 'gloo':
        raise RuntimeError('Post-backward CPU synchronization requires Gloo')
    for value in list(model.parameters()) + list(model.buffers()):
        host = value.detach().to(device='cpu', copy=True)
        dist.broadcast(host, src=0)
        value.copy_(host)


@torch.no_grad()
def average_gradients(model):
    """Preserve global unused=None and locally unused=zero DDP semantics."""
    if dist.get_backend() != 'gloo':
        raise RuntimeError('CPU gradient averaging requires Gloo')
    parameters = [p for p in model.parameters() if p.requires_grad]
    used = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32)
    dist.all_reduce(used)
    groups = {}
    for index, parameter in enumerate(parameters):
        if used[index].item() == 0:
            if parameter.grad is not None:
                raise RuntimeError('Inconsistent global gradient participation')
            continue
        if parameter.grad is not None and parameter.grad.is_sparse:
            raise RuntimeError('Sparse gradients are not supported by this candidate')
        groups.setdefault(parameter.dtype, []).append(parameter)
    world = dist.get_world_size()
    for dtype in sorted(groups, key=str):
        members = groups[dtype]
        host = torch.cat([
            p.grad.detach().to('cpu').reshape(-1) if p.grad is not None
            else torch.zeros(p.numel(), dtype=p.dtype) for p in members])
        host.div_(world)
        dist.all_reduce(host)
        offset = 0
        for parameter in members:
            if parameter.grad is None:
                parameter.grad = torch.empty_like(parameter)
            parameter.grad.copy_(host[offset:offset+parameter.numel()].view(parameter.shape))
            offset += parameter.numel()


def install_optimizer_hook(optimizer, model):
    """Runs before optimizer.step, including subclass gradient clipping."""
    return optimizer.register_step_pre_hook(lambda optimizer, args, kwargs: average_gradients(model))

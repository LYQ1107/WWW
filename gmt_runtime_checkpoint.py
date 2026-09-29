"""Recompute backbone activations; preserve original methods, parameters and BN state."""
import torch
from torch.utils.checkpoint import checkpoint


def copy_containers(value):
    if isinstance(value, list):return [copy_containers(x) for x in value]
    if isinstance(value, tuple):return tuple(copy_containers(x) for x in value)
    if isinstance(value, dict):return {k:copy_containers(v) for k,v in value.items()}
    return value


def wrap(module):
    if getattr(module, '_gmt_recompute_wrapped', False):raise RuntimeError('Already checkpointed')
    original = module.forward
    def forward(*args, **kwargs):
        if not module.training or not torch.is_grad_enabled():return original(*args, **kwargs)
        calls = 0
        def compute(*inputs):
            nonlocal calls
            recomputing = calls > 0
            calls += 1
            saved = [(b, b.detach().clone()) for m in module.modules()
                     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                     for b in (m.running_mean, m.running_var, m.num_batches_tracked)
                     if b is not None] if recomputing else []
            try:
                # BiFPN appends to input lists; copy containers on each invocation.
                return original(*copy_containers(inputs), **copy_containers(kwargs))
            finally:
                with torch.no_grad():
                    for buffer, value in saved:buffer.copy_(value)
        return checkpoint(compute, *args, use_reentrant=False, preserve_rng_state=True)
    module.forward = forward
    module._gmt_recompute_wrapped = True


def install(model):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    before = tuple(model.state_dict())
    backbone = model.backbone
    targets = [('backbone.bottom_up.base_layer', backbone.bottom_up.base_layer)]
    targets += [(f'backbone.bottom_up.level{i}', getattr(backbone.bottom_up, f'level{i}')) for i in range(6)]
    targets += [(f'backbone.cell.{i}', cell) for i, cell in enumerate(backbone.cell)]
    for name, module in targets:wrap(module)
    assert tuple(model.state_dict()) == before
    return [name for name, _ in targets]

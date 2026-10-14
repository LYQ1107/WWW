"""Deduplicate retained tensor/array storage; excludes model weights/allocator."""
import numpy as np
import torch


def retained_payload_bytes(value):
    objects, storages = set(), set()
    totals = dict(cpu_tensor_bytes=0, cuda_tensor_bytes=0, numpy_bytes=0)
    def visit(node):
        if id(node) in objects:
            return
        objects.add(id(node))
        if isinstance(node, torch.Tensor):
            storage = node.untyped_storage()
            key = (str(node.device), storage.data_ptr(), storage.nbytes())
            if key not in storages:
                storages.add(key)
                totals['cuda_tensor_bytes' if node.is_cuda else 'cpu_tensor_bytes'] += storage.nbytes()
        elif isinstance(node, np.ndarray):
            root = node
            while isinstance(root.base, np.ndarray):
                root = root.base
            key = ('numpy', root.__array_interface__['data'][0], root.nbytes)
            if key not in storages:
                storages.add(key)
                totals['numpy_bytes'] += root.nbytes
        elif isinstance(node, dict):
            for item in node.values():
                visit(item)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                visit(item)
        elif hasattr(node, 'get_fields'):
            visit(node.get_fields())
        elif hasattr(node, 'tensor'):
            visit(node.tensor)
    visit(value)
    totals['total_bytes'] = sum(totals.values())
    return totals

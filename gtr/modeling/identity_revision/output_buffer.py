"""Fixed scene-count publication window. Never retroactively rewrite release.

This implements publication ownership only. Any native identity revision must
also pass the controller/native transaction before updating pending payloads.
"""
from collections import OrderedDict
import copy


class OutputBuffer:
    def __init__(self,window=0):
        if window not in (0,2,4):raise ValueError('Only preregistered publication windows 0/2/4')
        self.window=window;self.reset()

    def reset(self):
        self.pending=OrderedDict();self.latest=None;self.sealed_through=None

    def push(self,timestamp,payload):
        if self.latest is not None and timestamp<=self.latest:raise ValueError('Nonmonotonic publication source')
        self.latest=timestamp;self.pending[timestamp]=copy.deepcopy(payload)
        release=[]
        while len(self.pending)>self.window:
            frame,value=self.pending.popitem(last=False);self.sealed_through=frame
            release.append((frame,copy.deepcopy(value)))
        return release

    def revise_pending(self,timestamp,payload):
        if timestamp not in self.pending:raise ValueError('Released or nonexistent frame is immutable')
        self.pending[timestamp]=copy.deepcopy(payload)

    def flush(self):
        release=[]
        while self.pending:
            frame,value=self.pending.popitem(last=False);self.sealed_through=frame
            release.append((frame,copy.deepcopy(value)))
        return release

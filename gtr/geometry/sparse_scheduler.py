"""Deterministic causal updates; all policies share a rolling call budget."""
from collections import deque
import numpy as np

class SparseScheduler:
    def __init__(self,policy='joint',interval=10,min_interval=2,max_age=20,budget=10,window=100,seed=20261014,offset=0):
        self.policy=policy;self.interval=interval;self.min_interval=min_interval;self.max_age=max_age;self.budget=budget;self.window=window;self.offset=offset
        self.calls=deque();self.last=None;self.rng=np.random.default_rng(seed)

    def should_update(self,t,motion=0.,margin=1.,propagation_residual=0.,handoff=False,budget_available=True):
        if self.last is not None and t<=self.last:raise ValueError('nonmonotonic input')
        while self.calls and self.calls[0]<=t-self.window:self.calls.popleft()
        age=float('inf') if self.last is None else t-self.last
        if not budget_available or len(self.calls)>=self.budget or age<self.min_interval:return False
        if self.policy=='every':want=True
        elif self.policy=='fixed':want=(t-self.offset)%self.interval==0
        elif self.policy=='random':want=self.rng.random()<1/self.interval
        elif self.policy=='motion':want=motion>=.08
        elif self.policy=='identity':want=margin<=.05 or handoff
        elif self.policy=='joint':want=motion>=.08 or margin<=.05 or handoff or age>=self.max_age or propagation_residual>=3.
        else:raise ValueError(self.policy)
        if want:self.last=t;self.calls.append(t)
        return bool(want)

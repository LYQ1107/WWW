"""Independent WHO and Q4 switches over the same actual committed raw Gallery."""
from gtr.modeling.jev_phase16.safe_matching_memory import SafeMatchingMemory
from gtr.modeling.jev_phase15.native_commit_adapter import CommitmentMemory
class FactorMemory(SafeMatchingMemory):
 def __init__(self,arm,prefix=None):
  assert arm in ['B0','B1','B2','B3'];super().__init__('bounded_quarantine',prefix);self.arm=arm;self.last_comparison=None
 def build(self,current,refs,galleries,frame,view,active,**kwargs):
  raw=CommitmentMemory.build(self,current,refs,galleries,frame,view,active,**kwargs)
  safe=SafeMatchingMemory.build(self,current,refs,galleries,frame,view,active,**kwargs)
  self.last_comparison={'raw':raw,'safe':safe,'refs':list(refs),'key':(frame,view)}
  result=dict(safe if self.arm in ['B1','B3'] else raw)
  result['commitment_features']=(safe if self.arm in ['B2','B3'] else raw)['commitment_features']
  return result

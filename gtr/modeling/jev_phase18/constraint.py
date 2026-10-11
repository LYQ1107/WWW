"""Lock one legal choice; solve all remaining rows with the exact native solver."""
from gtr.modeling.jev_stage2.assignment import lawful_choice

def constrained_choice(logits,legal,row,column):
 d,k=legal.shape;assert 0<=row<d;assert column is None or 0<=column<k and bool(legal[row,column])
 rows=[r for r in range(d) if r!=row];cols=[c for c in range(k) if c!=column];result=[-1]*d
 if column is not None:result[row]=column
 if rows:
  x=logits[rows][:,cols+[k]];mask=legal[rows][:,cols];other=lawful_choice(x,mask)
  for r,c in zip(rows,other):result[r]=cols[c] if c>=0 else -1
 positive=[c for c in result if c>=0];assert len(positive)==len(set(positive));assert all(c<0 or bool(legal[r,c]) for r,c in enumerate(result));assert result[row]==(-1 if column is None else column)
 return result

class NativeConstraint:
 """Scoped assignment-hook; scores and legal tensors are never mutated."""
 def __init__(self,executor,key,row,reference):self.executor=executor;self.key=tuple(key);self.row=row;self.reference=reference;self.fired=0;self.tensor=None;self.refs=None;self.task=None;self.audit=None
 def __enter__(self):
  import gtr.modeling.jev_stage2.native as native
  self.native=native;self.old_solver=native.lawful_choice;self.old_scores=self.executor.scores
  def scores(batch,refs,context,task):
   value=self.old_scores(batch,refs,context,task)
   if task==0 and (context['frame'],context['view'])==self.key and not self.fired:self.tensor=value;self.refs=refs
   return value
  def solve(logits,legal):
   if logits is not self.tensor or self.fired:return self.old_solver(logits,legal)
   assert self.reference is None or self.reference in self.refs
   column=None if self.reference is None else self.refs.index(self.reference);original=self.old_solver(logits,legal);actual=constrained_choice(logits,legal,self.row,column)
   self.fired+=1;self.audit=dict(row=self.row,forced_reference=self.reference,original_columns=original,constrained_columns=actual,refs=list(self.refs),changed_rows=[r for r,(a,b) in enumerate(zip(original,actual)) if a!=b],scores_unchanged=True,legal_masks_unchanged=True,second_scoring_validation_present=False,native_capacity_feasibility_verified=True)
   return actual
  self.executor.scores=scores;native.lawful_choice=solve;return self
 def __exit__(self,*args):self.executor.scores=self.old_scores;self.native.lawful_choice=self.old_solver

"""GT-free small legal alternatives; same proposals and budget for every policy."""
import torch
from gtr.modeling.jev_stage2.assignment import lawful_choice


def plans(logits, legal, maximum=4):
    assert maximum==4 and logits.shape==(legal.shape[0],legal.shape[1]+1)
    base=tuple(lawful_choice(logits,legal));result=[base]
    d,k=legal.shape
    if not d or not k:return result
    # Actual competing rows are identified from their legal current scores.
    values=logits.detach().double()
    competitors={}
    for row in range(d):
        cols=torch.where(legal[row])[0]
        if len(cols):
            top=int(cols[values[row,cols].argmax()])
            if values[row,top]>values[row,-1]:competitors.setdefault(top,[]).append(row)
    rows=sorted({r for group in competitors.values() if len(group)>1 for r in group})
    # Also allow an ambiguous, positive old-identity choice with another legal
    # near-scoring option; no GT, UNKNOWN label or future outcome enters this.
    for row,col in enumerate(base):
        if col<0:continue
        other=values[row].clone();other[col]=-1e12
        other[:k].masked_fill_(~legal[row],-1e12)
        if float(values[row,col]-other.max())<=.5:rows.append(row)
    ranked=sorted(set(rows),key=lambda r:(float(values[r,base[r]]-values[r,-1]) if base[r]>=0 else float('inf'),r))
    for row in ranked:
        col=base[row]
        if col<0:continue
        mask=legal.clone();mask[row,col]=False
        alternative=tuple(lawful_choice(logits,mask))
        if alternative not in result:result.append(alternative)
        if len(result)==maximum:break
    for plan in result:
        existing=[c for c in plan if c>=0]
        assert len(existing)==len(set(existing)), 'Capacity is per camera event'
        assert all(c<0 or bool(legal[r,c]) for r,c in enumerate(plan))
    return result


def selected_tokens(tokens, alternatives):
    q,kplus,d=tokens.shape
    indices=torch.tensor([[kplus-1 if col<0 else col for col in p] for p in alternatives],device=tokens.device)
    rows=torch.arange(q,device=tokens.device)[None].expand(len(alternatives),-1)
    return tokens[rows,indices],indices

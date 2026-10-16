"""Local visual evidence; mutual self-consistency is not physical geometry truth."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_tokens(first,second,temperature=.07):
    x=np.asarray(first,dtype=np.float32);y=np.asarray(second,dtype=np.float32)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1]:raise ValueError('token dimensions differ')
    m,n=len(x),len(y)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite local feature')
    nx=np.linalg.norm(x,axis=1);ny=np.linalg.norm(y,axis=1)
    if np.any(nx<=1e-12) or np.any(ny<=1e-12):raise ValueError('zero token is not valid evidence')
    result=dict(valid_tokens_A=m,valid_tokens_B=n,status='POINT_CORRESPONDENCE_UNAVAILABLE',
        evidence_type='VISUAL_CORRESPONDENCE_EVIDENCE',mutual_count=0,mutual_coverage_A=None,mutual_coverage_B=None,
        mean_mutual_similarity=None,hard_cycle_A=None,hard_cycle_B=None,soft_cycle_A=None,soft_cycle_B=None,
        best_second_margin=None,primary_score=None,cycle_weighted_score=None,ROI_cosine=None)
    if not m or not n:return result,dict(cosine=np.empty((m,n),np.float32))
    a=x/nx[:,None];b=y/ny[:,None];C=a@b.T
    ax=x.mean(0);by=y.mean(0);den=np.linalg.norm(ax)*np.linalg.norm(by)
    result['ROI_cosine']=float(ax@by/den) if den>1e-12 else None
    if min(m,n)<2:return result,dict(cosine=C)
    ab=C.argmax(1);ba=C.argmax(0);i=np.flatnonzero(ba[ab]==np.arange(m));j=ab[i]
    ca,cb=len(i)/m,len(i)/n;similarity=float(C[i,j].mean())
    def softmax(z):
        z=(z-z.max(1,keepdims=True))/temperature;v=np.exp(z);return v/v.sum(1,keepdims=True)
    P,Q=softmax(C),softmax(C.T)
    sa,sb=float(np.sum(P*Q.T)/m),float(np.sum(Q*P.T)/n)
    sorted_a=np.sort(C,axis=1);sorted_b=np.sort(C,axis=0)
    margin=float(.5*((sorted_a[:,-1]-sorted_a[:,-2]).mean()+(sorted_b[-1,:]-sorted_b[-2,:]).mean()))
    hi,hj=linear_sum_assignment(-C)
    primary=similarity*min(ca,cb)
    result.update(status='LOCAL_VISUAL_MATCH_MEASURED',mutual_count=len(i),mutual_coverage_A=ca,mutual_coverage_B=cb,
        mean_mutual_similarity=similarity,hard_cycle_A=ca,hard_cycle_B=cb,soft_cycle_A=sa,soft_cycle_B=sb,
        best_second_margin=margin,primary_score=primary,cycle_weighted_score=primary*min(sa,sb),
        NN_similarity_A=float(C[np.arange(m),ab].mean()),NN_similarity_B=float(C[ba,np.arange(n)].mean()),
        one_to_one_count=len(hi),one_to_one_similarity=float(C[hi,hj].mean()))
    return result,dict(cosine=C,mutual_indices_A=i,mutual_indices_B=j,nearest_B=ab,nearest_A=ba,
                       one_to_one_A=hi,one_to_one_B=hj)

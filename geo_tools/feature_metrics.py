"""Cross-camera retrieval, hard candidates, complementarity; no inference label access."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)

def evaluate_scores(items):
    ranks=[];hard_ranks=[];labels=[];scores=[];gaps=[];fixed=broken=bothwrong=total=0;query_ms=[]
    for item in items:
        score,ids1,ids2,base=item['score'],item['ids1'],item['ids2'],item['base']
        keep1=ids1>0;keep2=ids2>0;score=score[np.ix_(keep2,keep1)];base=base[np.ix_(keep2,keep1)];ids1=ids1[keep1];ids2=ids2[keep2]
        positive=ids2[:,None]==ids1[None,:]
        if score.size:
            labels.extend(positive.ravel());scores.extend(score.ravel())
        for i in range(len(ids2)):
            if not positive[i].any():continue
            order=np.argsort(-score[i],kind='stable');rank=int(np.flatnonzero(positive[i,order])[0])+1;ranks.append(rank)
            base_order=np.argsort(-base[i],kind='stable');base_ok=positive[i,base_order[0]];new_ok=rank==1
            fixed+=int(new_ok and not base_ok);broken+=int(base_ok and not new_ok);bothwrong+=int(not base_ok and not new_ok);total+=1
            positives=score[i,positive[i]];negatives=score[i,~positive[i]]
            if len(negatives):gaps.append(float(positives.max()-negatives.max()))
            if (~positive[i]).any() and (base[i,~positive[i]].max()>=base[i,positive[i]].max()-.05):hard_ranks.append(rank)
    ranks=np.asarray(ranks);labels=np.asarray(labels);scores=np.asarray(scores)
    has_auc=len(np.unique(labels))==2
    return dict(queries=len(ranks),recall1=float(np.mean(ranks<=1)) if len(ranks) else None,recall5=float(np.mean(ranks<=5)) if len(ranks) else None,
        mean_rank=float(np.mean(ranks)) if len(ranks) else None,mrr=float(np.mean(1/ranks)) if len(ranks) else None,
        similarity_margin=float(np.mean(gaps)) if gaps else None,hard_queries=len(hard_ranks),hard_recall1=float(np.mean(np.array(hard_ranks)<=1)) if hard_ranks else None,
        pair_auc=float(roc_auc_score(labels,scores)) if has_auc else None,pair_ap=float(average_precision_score(labels,scores)) if has_auc else None,
        fixes_gmt_top1=fixed,breaks_gmt_top1=broken,both_wrong=bothwrong,pair_count=len(labels))

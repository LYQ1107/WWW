"""Compact all 51 preregistered fits; temporally clustered paired diagnostics."""
import collections,numpy as np
from jev_phase12_common import *

def main():
    protect();protocol=json.loads((REPORTS/'MATCH_TRAINING_PROTOCOL.json').read_text());results={};curves=[]
    metrics=['CertifiedCorrect','CertifiedWrong','UNKNOWNChoice','rank1','MRR','known_NLL','known_Brier','known_ECE10','executed_H32_regret','executed_pair_order_accuracy']
    for name in protocol['models']:
        seeds=[]
        for seed in protocol['seeds']:
            path=OUT/'formal_training_v1'/name/'joint'/f'seed{seed}'/'RESULT.json';r=json.loads(path.read_text());assert r['status']=='COMPLETE_STABLE'
            assert r['actual_updates']==900 and r['gradient_health']['finite'];assert r['binding']['source_commit']=='2e00194cf159e9757c6ce085a17a1bb3a4912102'
            results[name,seed]=r
            seeds.append({k:r[k] for k in ['seed','capacity','best_epoch','temperature','actual_updates','gradient_health','checkpoints','binding','elapsed_seconds']} | {split:{m:r[split][m] for m in metrics} for split in ['TRAIN','VALIDATION']})
            curves.append({'model':name,'seed':seed,'curve':[{'update':i['update'],'selection_criterion':i['selection_criterion'],'train_correct':i['TRAIN']['CertifiedCorrect'],'validation_correct':i['VALIDATION']['CertifiedCorrect']} for i in r['history']]})
        means={m:float(np.mean([s['VALIDATION'][m] for s in seeds])) if all(s['VALIDATION'][m] is not None for s in seeds) else None for m in metrics}
        results[name]={'seeds':seeds,'VALIDATION_mean':means}
    paired={};rng=np.random.default_rng(20261009)
    for baseline in [n for n in protocol['models'] if n!='full']:
        values=collections.defaultdict(list)
        for seed in protocol['seeds']:
            a=results['full',seed]['VALIDATION']['records'];b=results[baseline,seed]['VALIDATION']['records']
            assert [(r['key'],r['row'],r['group'],r['bundle'],r['weight']) for r in a]==[(r['key'],r['row'],r['group'],r['bundle'],r['weight']) for r in b]
            for x,y in zip(a,b):values[x['bundle']].append((x['weight'],float(x['certificate']=='CORRECT')-float(y['certificate']=='CORRECT')))
        bundles=sorted(values);sums=np.array([[sum(w*d for w,d in values[b]),sum(w for w,d in values[b])] for b in bundles]);boot=[]
        for _ in range(5000):
            total=sums[rng.integers(0,len(sums),len(sums))].sum(0);boot.append(total[0]/total[1])
        delta=float(sums[:,0].sum()/sums[:,1].sum());ci=list(map(float,np.quantile(boot,[.025,.975])))
        paired[baseline]={'full_minus_baseline_certified_correct':delta,'temporal_bundle_bootstrap95':ci,'bundles':bundles,'positive_lower_bound':ci[0]>0,'scope':'paired lower-bound certificate metric on partial labels, 3 coupled seeds; few independent validation videos; diagnostic CI, not confirmatory test or complete identity accuracy'}
    report={'status':'COMPLETE_STABLE','fits':51,'scope':'offline native v3 development validation, NOT closed-loop MOT','protocol_SHA256':sha(REPORTS/'MATCH_TRAINING_PROTOCOL.json'),'dataset_SHA256':results['full',20261008]['visual_dataset_SHA256'],'source_commit':results['full',20261008]['binding']['source_commit'],'models':{name:results[name] for name in protocol['models']},'paired_diagnostics':paired,'architecture_redesigned_after_validation':False,'UNKNOWN_is_negative':False,'unexecuted_Q_imputed':False,'heldout':'SEALED'}
    save(REPORTS/'MATCH_OFFLINE_VALIDATION.json',report);save(REPORTS/'MATCH_TRAINING_CURVES.json',{'status':'COMPLETE','curves':curves})
    save(REPORTS/'ARCHITECTURE_ABLATION.json',report | {'same_visual_models':protocol['same_visual_main'],'same_capacity_visual_removal':'numerical_only','no_risk':'same fitted Full checkpoint; online execution intervention, pending','inactive_capacity_notes':{'no_consequence':'registered head still computes; loss/inference contribution removed, not computation pruning','numerical_only':'full parameter count; visual tensors zero, inactive visual pathways disclosed'}})
    print('ALL51_FITS_PUBLISHED',flush=True)
if __name__=='__main__':main()

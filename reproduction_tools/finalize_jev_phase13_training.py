"""Aggregate actual outputs and gate the next optimization stage."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jev_phase13_learning import *

def main(phase):
    protect();protocol=json.loads((REPORTS/'TRAINING_PROTOCOL.json').read_text());raw=[]
    cases=[('full',s) for s in SEEDS] if phase=='tiny' else [(v,20261009) for v in ['full','motip','camel','set_transformer']] if phase=='pilot' else [(v,s) for v in VARIANTS for s in SEEDS]
    for variant,seed in cases:
        p=OUT/f'{phase}_training_v1'/variant/f'seed{seed}'/'RESULT.json';d=json.loads(p.read_text());assert d['status']=='COMPLETE' and d['actual_updates']==protocol['updates'][phase];assert sha(d['checkpoint']['path'])==d['checkpoint']['SHA256'];raw.append({'variant':variant,'seed':seed,'path':str(p),'SHA256':sha(p),'result':d})
    if phase=='tiny':
        records,_=load_data(TRAIN);keys={tuple(k) for k in protocol['tiny_record_keys']};rows=[r for r in records if tuple(r['key'])+(r['task'],) in keys];per=[]
        for item in raw:
            d=item['result'];model=GlobalIdentityJev().cuda().eval();model.load_state_dict(torch.load(d['checkpoint']['path'],map_location='cpu')['model']);task={str(t):evaluate(model,[r for r in rows if r['task']==t]) for t in [0,1]};per.append({'seed':d['seed'],'by_task':task})
        passed=all(p['by_task']['0']['certified_accuracy']>=protocol['tiny_gate']['MATCH_certified_accuracy_min'] for p in per)
        report={'status':'PASS' if passed else 'FAIL','binding':binding(),'results':raw,'actual_per_task':per,'frozen_selection':protocol['tiny_selection_categories'],'Tiny_RECOVERY_only_memorization_not_qualification':True,'formal_REACT':'UNTRAINED_FROZEN_FALLBACK','formal_MEMORY':'UNTRAINED_FROZEN_FALLBACK','scope':'real frozen TRAIN examples only; no tracking/generalization claim'};name='TINY_LEARNABILITY'
    elif phase=='pilot':
        full=next(r['result'] for r in raw if r['variant']=='full');passed=full['TRAIN_FIXED_AUDIT_SUBSET']['certified_accuracy']>=protocol['pilot_gate']['normal_matching_certified_accuracy_min'] and all(r['result']['gradient_health']['finite'] for r in raw)
        report={'status':'PASS' if passed else 'FAIL','binding':binding(),'results':raw,'updates_per_second':{r['variant']:r['result']['actual_updates']/r['result']['elapsed_seconds'] for r in raw},'formal_budget':'20000 fixed optimizer updates, fresh initialization, all variants/seeds','formal_architecture_hyperparameter_changes':False,'GTA_free_dense_pilot_complete':True};name='PILOT'
    else:
        report={'status':'COMPLETE','binding':binding(),'results':raw,'models':VARIANTS,'seeds':SEEDS,'updates':20000,'primary_checkpoint':'LAST at common budget','lifecycle_heads':{'MATCH':'TRAINED','REACT':'UNTRAINED_FALLBACK','MEMORY':'UNTRAINED_FALLBACK'},'HOTA_checkpoint_selection':False};name='SUPERVISED_TRAINING'
    save(REPORTS/(name+'.json'),report)
    fig,ax=plt.subplots(figsize=(7,4))
    for item in raw:
        d=item['result'];h=d.get('history')
        if h is None:h=torch.load(d['checkpoint']['path'],map_location='cpu')['history']
        ax.plot([v['update'] for v in h],[v['DEVELOPMENT']['known_NLL'] for v in h],label=f"{d['variant']} {d['seed']}")
    ax.set(xlabel='Optimizer updates',ylabel='Certified-option NLL',title=f'Phase XIII {phase}: actual saved training curves');ax.set_yscale('log');ax.grid(alpha=.2)
    if len(raw)<=12:ax.legend(fontsize=6,ncol=2)
    fig.tight_layout();path=REPORTS/(name+'_CURVE.png');fig.savefig(path,dpi=150);plt.close(fig);print('PHASE13_TRAIN_AGGREGATE',phase,report['status'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tiny','pilot','formal'],required=True);main(p.parse_args().phase)

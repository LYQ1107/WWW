"""Predeclared TRAIN-only mechanism gate; no DEV selection, no synthetic completion."""
from jev_recovery_common import *
def evaluate(revision,base):
 arms=['jev','threshold','mlp','attention'];training={a:read(REPORTS/f'TRAIN_R{revision}_{base}_{a}_pilot_SEED20261009.json') for a in arms}
 root=OUT/f'R{revision}_{base}'/'online_v1/pilot';native={a:read(root/a/'seed20261009/video16/RESULT.json') for a in ['parent',*arms]}
 # Independent cold trajectories: official identity/MOT before/after, not paired causal effects.
 rows={}
 parent=native['parent']['identity_summary'];base_wrong=parent.get('counts',{}).get('owner_incompatible',None)
 for arm in arms:
  t=training[arm];before=t['before']['known_correct_rate'];after=t['final']['known_correct_rate']
  m=native[arm]['raw_online_metrics'];pm=native['parent']['raw_online_metrics']
  # Confirm measurable current-association evidence and no severe TRAIN16 MOT collapse.
  hota=m.get('HOTA');phota=pm.get('HOTA')
  rows[arm]=dict(ENGINEERING='PASS' if t['engineering_training_pass'] else 'FAIL',SUPERVISION='CURRENT_MATCH_ELIGIBLE',controller_known_correct_delta=after-before,rank_nonzero_batch_fraction=t['rank_nonzero_batch_fraction'],TRAIN16_HOTA=hota,TRAIN16_AssA=m.get('AssA'),TRAIN16_IDSW=m.get('IDSW'),TRAIN16_HOTA_delta=hota-phota if hota is not None and phota is not None else None,current_mechanism=(after>before and t['rank_supervised_batches']>0),no_severe_online_collapse=hota is not None and phota is not None and hota>=phota-1.)
 eligible=any(r['current_mechanism'] and r['no_severe_online_collapse'] for r in rows.values())
 value=dict(status='COMPLETE_TRAIN_GATE',revision=revision,base=base,rows=rows,formal_eligible=eligible,JEV_independent_value='NOT_ESTABLISHED',EFFICIENCY='PENDING',CROSS_CAMERA='NOT_ESTABLISHED',ASSOCIATION='TRAIN_DIAGNOSTIC_ONLY',rule='At least one controlled arm improves deterministic known TRAIN16 choices after training and has nonzero certified ranks; cold TRAIN16 HOTA may fall at most 1 point before formal promotion. Gate identical across arms; none selected by DEV.',independent_validation=False,master_complete=False,GT_only_posthoc=True)
 save(REPORTS/f'R{revision}_{base}_MECHANISM_GATE.json',value);return value

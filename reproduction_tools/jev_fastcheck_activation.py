"""Evaluate the frozen conditional P3 rule once on complete P2 results."""
from jev_fastcheck_common import *
def main():
 rows={tag:read(REPORTS/f'ONLINE_{tag}_STEPS1000.json') for tag in ['attention_D0','attention_D1','jev_D0','jev_D1']}
 for r in rows.values():assert r['status']=='COMPLETE' and r['actual_mutated_state'] and r['frames']==1034
 j0,j1=rows['jev_D0'],rows['jev_D1'];a0,a1=rows['attention_D0'],rows['attention_D1']
 delta={k:j1['raw_online_metrics'][k]-j0['raw_online_metrics'][k] for k in ['HOTA','AssA','IDF1','IDSW','MOTA']}
 delta['CVIDF1']=j1['CVIDF1']-j0['CVIDF1']
 checks=dict(HOTA_plus_point5=delta['HOTA']>=.5,AssA_positive=delta['AssA']>0,CVIDF1_no_drop=delta['CVIDF1']>=0,IDSW_within_5pct=j1['raw_online_metrics']['IDSW']<=1.05*j0['raw_online_metrics']['IDSW'])
 result=dict(status='COMPLETE_PREDECLARED_ACTIVATION_GATE',execute_P3=not all(checks.values()),checks=checks,J1_minus_J0=delta,
  A1_minus_A0={k:a1['raw_online_metrics'][k]-a0['raw_online_metrics'][k] for k in delta if k!='CVIDF1'},
  evidence=[ref(REPORTS/f'ONLINE_{tag}_STEPS1000.json') for tag in rows],protocol=ref(REPORTS/'P3_CONDITIONAL_PREREGISTRATION.json'),no_original_GATE_changed=True,independent_validation=False)
 save(REPORTS/'P3_ACTIVATION_GATE.json',result)
 print('P3_ACTIVATION',result['execute_P3'],delta,checks,flush=True)
if __name__=='__main__':main()

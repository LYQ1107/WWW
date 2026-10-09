"""Independent Tiny evidence; stable numerics permits full v3 development."""
from jev_phase12_common import *

def main():
    protect();runs=[]
    for supervision in ['CE','H32','joint']:
        for seed in [20261008,20261009,20261010]:
            p=OUT/'tiny_training_v1/full'/supervision/f'seed{seed}'/'RESULT.json'
            assert p.exists(),'wait for all predeclared Tiny runs'
            d=json.loads(p.read_text());assert d['status']=='COMPLETE_STABLE' and d['actual_updates']==1000 and d['gradient_health']['finite']
            assert d['TRAIN']['rows']==12
            for checkpoint in d['checkpoints'].values():assert sha(checkpoint['path'])==checkpoint['SHA256']
            runs.append({k:v for k,v in d.items() if k not in ['binding','VALIDATION']}|{'binding':d['binding'],'raw_result_path':str(p),'raw_result_SHA256':sha(p),'fit_set':'frozen Tiny12 rows/eight groups only','supervision':supervision,'seed':seed})
    result={'status':'COMPLETE_NUMERICALLY_STABLE','runs':runs,'tiny_rows':12,'groups':8,'old_PhaseX_Tiny_FAIL_unchanged':True,'all_models95_Tiny_gate_required':False,'full_MATCH_training_permitted':True,'reason':'all native/data/structural/supervision gates passed; all nine new Tiny fits have finite losses and gradients; certification failures remain explicit','not_real_online_MOT_evidence':True,'heldout_status':'SEALED','Full24':False,'official_TEST':False}
    save(REPORTS/'MATCH_TINY_RESULTS.json',result);print('TINY_STABLE_NINE_RUNS_FULL_V3_PERMITTED',flush=True)

if __name__=='__main__':main()

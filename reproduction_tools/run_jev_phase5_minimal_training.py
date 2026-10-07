"""Two fixed single-seed MATCH-only controls; val-only calibration."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2'):sys.path.insert(0,str(p))
from audit_jev_phase5 import OUT,BASE,records,sha
from run_segmented_small_gate import atomic,PYTHON


def prepare_control():
    from jev_compact_dataset import build_compact_dataset
    root=OUT/'match_training_legacy';paths=[]
    for v in (6,7):
        path=root/f'video{v:02d}_legacy_match.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        rows=[r for r in records(v) if r['question_type']=='MATCH_DECISION'];path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));paths.append(path)
    build_compact_dataset(paths,root/'compact')
    split=json.loads((BASE/'small_h8_training_current_head_video06_video07/policy_split.json').read_text());split.update(source_files=[str(p) for p in paths],source_sha256={str(p):'sha256:'+sha(p) for p in paths},source_record_count=4303,train_record_count=1703,val_record_count=2600,purpose='phase5_minimal_match_only_legacy_coordinate_control')
    atomic(root/'policy_split.json',split)
    atomic(OUT/'MINIMAL_TRAINING_PLAN.json',{'status':'PREDECLARED','conditions':{'B1':'MATCH-only legacy-coordinate diagnostic control; not promotion eligible','B2':'MATCH-only corrected-frame labels, complete video07 train / video06 val'},'model':'jev','hidden_dim':128,'seed':20261003,'epochs':20,'batch_size':128,'optimizer':'AdamW','lr':0.001,'checkpoint_selection':'fixed last epoch20; no held-out tuning','temperature_fit':'video06 MATCH validation only','frozen_A1_retained':True,'official_test_read':False,'selection_video01_is_reused_diagnostic_not_independent_test':True,'heldout_control_routing':'JEV MATCH; GMT MEMORY and REACTIVATION','predeclared_input_hashes':{'B1_compact':sha(root/'compact/manifest.json'),'B1_split':sha(root/'policy_split.json')}})
    print(json.dumps({'status':'PREPARED','B1_records':4303}))


def train(condition):
    root=OUT/('match_training_legacy' if condition=='B1' else 'match_training');out=OUT/'minimal_training'/condition;out.mkdir(parents=True,exist_ok=True)
    if (out/'model.pth').exists():raise RuntimeError('refusing to overwrite trained checkpoint')
    atomic(out/'binding.json',{'condition':condition,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'compact_manifest_sha256':sha(root/'compact/manifest.json'),'split_sha256':sha(root/'policy_split.json'),'official_test_read':False})
    commands=[
        [PYTHON,'-u',str(ROOT/'reproduction_tools/train_jev_compact.py'),'--dataset',str(root/'compact'),'--output',str(out),'--model','jev','--epochs','20','--batch-size','128','--hidden-dim','128','--lr','0.001','--seed','20261003','--device','cuda:0','--split-manifest',str(root/'policy_split.json')],
        [PYTHON,'-u',str(ROOT/'reproduction_tools/calibrate_jev_compact.py'),'--dataset',str(root/'compact'),'--checkpoint',str(out/'model.pth'),'--policy-split',str(root/'policy_split.json'),'--output',str(out/'calibration')]]
    for phase,command in zip(('TRAINING','CALIBRATION'),commands):
        atomic(out/'runtime_status.json',{'phase':phase,'condition':condition,'command':command,'pid':os.getpid()})
        subprocess.run(command,cwd=ROOT,check=True)
    atomic(out/'runtime_status.json',{'phase':'COMPLETE','condition':condition,'checkpoint_sha256':sha(out/'calibration/model_calibrated.pth')})
    print(json.dumps({'status':'COMPLETE','condition':condition}))


def aggregate():
    results={};off=json.loads((OUT/'ablations/A0/result.json').read_text())['metrics'];frozen=json.loads((OUT/'ablations/A1/result.json').read_text())['metrics']
    for condition in ('B1','B2'):
        r=json.loads((OUT/'minimal_tracking'/condition/'result.json').read_text());training=json.loads((OUT/'minimal_training'/condition/'metrics.json').read_text());calibration=json.loads((OUT/'minimal_training'/condition/'calibration/calibration_val_only.json').read_text())
        results[condition]={'tracking':r,'training':training,'calibration':calibration,'delta_vs_OFF':{k:r['metrics'][k]-off[k] for k in off},'delta_vs_frozen_A1':{k:r['metrics'][k]-frozen[k] for k in frozen},'above_OFF_both_AssA_HOTA':all(r['metrics'][k]>off[k] for k in ('AssA','HOTA'))}
    value={'status':'COMPLETE','conditions':results,'fixed_original_MATCH_controller_retained':True,'no_checkpoint_selected_by_heldout_video01':True,'official_test_read':False,'interpretation_scope':'single-seed TRAIN diagnostic; B1/B2 isolate coordinate correction within the same MATCH-only objective; old full-question checkpoint A1 is the frozen reference'}
    atomic(OUT/'MINIMAL_RETRAINING.json',value);print(json.dumps({c:d['tracking']['metrics'] for c,d in results.items()}))

if __name__=='__main__':
    if sys.argv[1]=='train':train(sys.argv[2])
    else:globals()[sys.argv[1]]()

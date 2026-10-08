"""Per-run reproduction bundles without rewriting original completed results."""
import json,subprocess
from jev_phase7_common import *

def main():
    protect();data=json.loads((REPORTS/'HELDOUT_RESULTS.json').read_text())['results']
    environment=json.loads((REPORTS/'EXPERIMENT_ENVIRONMENT.json').read_text());records=[]
    checkpoints={'B2':B2,'BINARY':Path('/home/liuyeqiang/WWW_jev_phase6_runtime/20261008/gating_training/G3/model_calibrated.pth'),
        'MLP':OUT/'training/MLP/model_calibrated.pth','DYNAMIC':OUT/'training/DYNAMIC/model_calibrated.pth',
        'MLP_LN':OUT/'normalization_control/training/MLP/model_calibrated.pth'}
    checkpoint_info={name:{'path':str(path),'sha256':sha(path)}for name,path in checkpoints.items()}
    for tag,summary in data.items():
        actor=next((n for n in ['MLP_LN','BINARY','DYNAMIC','MLP','B2']if tag.startswith(n)),None)
        for run in summary['runs']:
            original=Path(run['predictions']).parent/'result.json'if tag.startswith('BYTETRACK')else OUT/'closed_loop'/f'video{run["video"]:02d}'/tag/'result.json'
            bind=run['binding'];head=bind['git_commit'];file_checks={}
            for file,digest in bind['source_sha256'].items():
                try:content=subprocess.check_output(['git','show',head+':'+file],cwd=ROOT,stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError:content=None
                import hashlib
                file_checks[file]={'recorded_sha256':digest,'equals_recorded_HEAD_file':content is not None and hashlib.sha256(content).hexdigest()==digest}
            record={'tag':tag,'video':run['video'],'input_split':'TRAIN','controller_heldout':True,
                'original_result_path':str(original),'original_result_sha256':sha(original),'original_binding':bind,
                'source_inventory_at_result_time':file_checks,'source_scope':'inventory may include unused new scripts; a HEAD mismatch is disclosed rather than called a clean execution revision',
                'worktree_dirty_at_original_capture':bind.get('git_worktree_dirty','UNKNOWN_NOT_RECORDED'),
                'actor_checkpoint':checkpoint_info.get(actor),
                'second_validator_checkpoint':checkpoint_info.get(actor)if run.get('validator')=='OWN'and actor else None,
                'second_validator_kind':run.get('validator','OFFICIAL_BYTETRACK_CASCADE'),
                'foundation_checkpoint':{'path':environment['foundation_checkpoint'],'sha256':environment['foundation_sha256']},
                'config_sha256':environment['config_sha256'],'seed':bind['seed'],'environment':environment,
                'perception_cache_version':environment['cache_version'],'cache_index_sha256':bind['cache_index_sha256'],
                'annotation_sha256':bind['annotations_sha256'],'initial_state_fingerprint':run.get('initial_state_fingerprint'),
                'GPU_assignment_source':'EXECUTION_PLAN_PRIMARY/CONTROLS/BUDGET/SUPPLEMENTS and named raw run logs; primary normalized supplement uses GPUs4/5/6',
                'official_test_read':False}
            if tag.startswith('BYTETRACK'):
                record['parameters']=run['parameters'];record['official_source_sha256']=run['official_source_sha256'];record['official_tracker_commit']=run['official_tracker_commit']
            records.append(record)
    manifest={'status':'COMPLETE_SUPPLEMENTAL_PER_RUN_REPRODUCTION_BINDINGS','runs':records,'count':len(records),
        'original_completed_results_changed':False,'checkpoint_inventory':checkpoint_info,
        'execution_start_plans':{p.name:sha(p)for p in REPORTS.glob('EXECUTION_PLAN_*.json')},
        'runtime_source_history':'training/inference source hashes are retained; HEAD-only reproducibility is not claimed where pending files or later publication commits were present',
        'binding':binding()}
    save(REPORTS/'RUN_REPRODUCTION_MANIFESTS.json',manifest);protect();print(json.dumps({'status':manifest['status'],'runs':len(records)}))

if __name__=='__main__':main()

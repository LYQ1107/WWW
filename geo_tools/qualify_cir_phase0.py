"""Close the engineering substage only; never declares the research Goal achieved."""
import argparse
import json
from pathlib import Path
import shutil
from common import ROOT,sha256,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True)
    p.add_argument('--official',type=Path,required=True);a=p.parse_args()
    native=json.loads(a.native.read_text());official=json.loads(a.official.read_text())
    if native['status']!='PASS_REAL_PREFIX_OFF_KEEP_ROLLBACK' or not native['full_video'] or native['groups']!=901:
        raise ValueError('Actual complete original/OFF/KEEP cold start required')
    if native['scene']!=4 or not all(all(c.values()) for c in native['comparisons'].values()):raise ValueError('Full native mismatch')
    if not all(native['comparisons'][arm].get('decisions') for arm in ('CIR_OFF','CIR_KEEP')):raise ValueError('Full joint score/solver record equality required')
    if official['status']!='PASS_REAL_PREFIX_OFFICIAL_CAUSAL' or not all(all(c.values()) for c in official['comparisons'].values()):raise ValueError('Official shared-core parity required')
    sources={}
    for name in ('gtr/config.py','gtr/modeling/meta_arch/gtr_rcnn.py','gtr/geometry/online_runner.py',
                 'gtr/modeling/identity_revision/controller.py'):
        digest=sha256(ROOT/name)
        if native['source_sha256_at_import'][name]!=digest or official['source_sha256_at_import'][name]!=digest:
            raise ValueError('Core changed after actual equivalence experiment '+name)
        sources[name]=digest
    component=json.loads((ROOT/'results/CIR_GMT/P0_COMPONENT_CONTRACTS.json').read_text())
    if component['tests_passed']!=41 or component['status']!='PASS_COMPONENT_CONTRACTS_FULL_VIDEO_QUALIFICATION_PENDING':raise ValueError('Bounded component contracts missing')
    for name,digest in component['test_sources'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Component contract source changed '+name)
    parser=json.loads((ROOT/'results/CIR_GMT/P0_METRIC_PARSER_SMOKE.json').read_text())
    if parser['status']!='PASS_SYNTHETIC_OFFICIAL_PARSER_SMOKE' or parser['scientific_result']:raise ValueError('Official metric parsing not qualified')
    result=dict(status='PASS_FULL_VIDEO_NATIVE_OFF_QUALIFICATION',scientific_success=False,
        current_source_sha256=sources,native_result=dict(path=str(a.native),sha256=sha256(a.native)),
        official_result=dict(path=str(a.official),sha256=sha256(a.official)),
        scope='TRAIN04 all901 synchronized groups, original source/OFF/KEEP, all predictions/native state/RNG/joint costs+decisions; actual branch restore; separate official shared-core40 prefix',
        whole_video_official_causal_comparison='pending independent streaming official driver check',
        C1_or_Guard_native_qualified=False,SPLIT_RELINK_qualified=False,
        actual_training_videos=0,official_TEST_executed=False,tracking_metrics=None)
    save_json(ROOT/'results/CIR_GMT/P0_NATIVE_QUALIFICATION.json',result)
    shutil.copyfile(a.native,ROOT/'results/CIR_GMT/P0_FULL_NATIVE_PARITY.json')
    save_json(ROOT/'results/CIR_GMT/BASELINE_OFF.json',dict(status='PASS_NATIVE_OFF_ENGINEERING_METRICS_PENDING',
        video=4,full_video=True,images=1802,methods=['original_source','CIR_OFF','CIR_KEEP'],
        exact_predictions=True,exact_native_states=True,exact_RNG=True,exact_joint_scores_and_decisions=True,
        raw={k:None for k in ('HOTA','AssA','CVIDF1','IDF1','IDSW','DetA','MOTA','Frag','FPS')},
        null_reason='Profiling includes heavy audit hashes; formal complete-image evaluation driver not yet run. No historical scores substituted.',
        source_result_sha256=sha256(a.native)))
    goal_path=ROOT/'results/CIR_GMT/RESEARCH_GOAL.json';goal=json.loads(goal_path.read_text())
    goal['phases']['0']='COMPLETE_NATIVE_OFF_KEEP_QUALIFIED';goal['phases']['1']='IN_PROGRESS_OFFICIAL_DRIVER_THEN_C1'
    goal['phase0_substates']['full_video_OFF_KEEP']='PASS_TRAIN04_ALL901_GROUPS_WITH_SCORE_STATE_RNG_EQUALITY'
    goal['status']='ACTIVE';goal['scientific_success']=False;save_json(goal_path,goal)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

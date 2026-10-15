"""Prospective integration of the measured B7 tracking candidate into retrieval."""
from common import *


def main():
    stages=[]
    def stage(identifier,command,cpu=False,depends=()):
        stages.append(dict(id=identifier,command=command,cpu_only=cpu,depends_on=list(depends)))
    prefix='retrieval_tracking_vggt_dense_4streams_prefix40_v1'
    stage(prefix,['geo_tools/run_live_retrieval.py','--video-ids','18','1','--tag',prefix,
                  '--backbone','none','--tracking-vggt-dense','--max-frames','40'])
    gate='vggt_tracking_retrieval_prefix_gate_v1'
    stage(gate,['geo_tools/audit_tracking_retrieval_prefix.py','--runs',str(RUNTIME/'retrieval'/prefix),
        '--reference-sources','manifests/causal_B7_fixed_k1_strict_suite_v1_sources.json',
        '--report-output',str(RUNTIME/'matrix_reports'/f'{gate}.json')],True,[prefix])
    for scenes in ([18],[18,1]):
        tag=f'retrieval_vggt_dense_tracking_{2*len(scenes)}streams_full_v1'
        stage(tag,['geo_tools/run_live_retrieval.py','--video-ids',*map(str,scenes),'--tag',tag,
                  '--backbone','none','--tracking-vggt-dense'],False,[gate])
        stage('eval_'+tag,['geo_tools/evaluate_live_retrieval.py','--run',str(RUNTIME/'retrieval'/tag)],True,[tag])
    save_json(ROOT/'manifests/vggt_tracking_retrieval_v1.json',dict(id='vggt_tracking_retrieval_v1',project_gpu_ids=['4'],queues={'gpu4':stages},
        scope='primary-seed complete-dev-positive dense B7 tracker feeding the unchanged native-ranked clip index; native extra-feature ranker disabled; all VGGT computation charged',
        gate='actual interleaved two-scene prefix equality to standalone B7, including exact tracking settings/head hashes; frozen encoder/head shared serially, scene scheduler/geometry/history/RNG separate',
        scale_scope='complete two/four-camera dense candidate; compare to corresponding OFF and fixed-prior controls; ten-camera OFF/sparse-feature and fixed-prior matrix remains separate, not extrapolated to dense tracking',provenance=provenance()))


if __name__=='__main__':main()

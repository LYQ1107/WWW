"""Predeclare fixed-prior tracking publication controls, gated by real prefixes."""
from common import *


def main():
    stages=[]
    def stage(identifier,command,cpu=False,depends=()):
        stages.append(dict(id=identifier,command=command,cpu_only=cpu,depends_on=list(depends)))
    prefixes=[]
    for prior in (False,True):
        tag=f"retrieval_tracking_{'prior' if prior else 'off'}_4streams_prefix40_v1"
        command=['geo_tools/run_live_retrieval.py','--video-ids','18','1','--tag',tag,'--backbone','none','--max-frames','40']
        if prior:command.append('--tracking-native-prior')
        stage(tag,command);prefixes.append(tag)
    gate='tracking_retrieval_prefix_gate_v1'
    stage(gate,['geo_tools/audit_tracking_retrieval_prefix.py','--runs',*[str(RUNTIME/'retrieval'/tag) for tag in prefixes],
        '--reference-sources','manifests/causal_B0_strict_suite_v1_sources.json','manifests/native_prior_seed20261014_strict_suite_v1_sources.json',
        '--report-output',str(RUNTIME/'matrix_reports'/f'{gate}.json')],True,prefixes)
    for scenes in ([18],[18,1],[18,1,10,11,19]):
        tag=f'retrieval_native_prior_{2*len(scenes)}streams_full_v1'
        stage(tag,['geo_tools/run_live_retrieval.py','--video-ids',*map(str,scenes),'--tag',tag,'--backbone','none','--tracking-native-prior'],False,[gate])
        stage('eval_'+tag,['geo_tools/evaluate_live_retrieval.py','--run',str(RUNTIME/'retrieval'/tag)],True,[tag])
    save_json(ROOT/'manifests/tracking_retrieval_control_v1.json',dict(id='tracking_retrieval_control_v1',project_gpu_ids=['4'],queues={'gpu4':stages},
        scope='explicit ordinary fixed-prior control feeding an unchanged native-ranked causal clip index; not VGGT-specific contribution or final candidate selection',
        gate='exact OFF and fixed-prior standalone identity prefixes on two interleaved scenes; no added tracking vectors used for native retrieval',provenance=provenance()))


if __name__=='__main__':main()

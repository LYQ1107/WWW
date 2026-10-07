"""Real native B2 prefix regression through first four reassociation events."""
import copy
import json
from pathlib import Path

from jev_phase6_common import OUT,save
from jev_phase6_rollouts import NativeReplayLab


def main(device):
    lab=NativeReplayLab(1,device,native_match_validation=True,record_transitions=True)
    lab.keys=[k for k in lab.keys if k[1]<=240]
    selected=[]
    def before(key,payload,state,kwargs,rows):
        for row in rows:
            if row['question']=='MEMORY_DECISION' and len(selected)<3:
                c=row['context'];selected.append({'key':[1,c['frame'],c['view'],row['question'],c['detection_index']],
                    'track_id':c['track_id'],'read_expected':False})
    result=lab.run(OUT/'smoke/native_journal',before_step=before)
    expected=json.loads((OUT/'native_b2/video01/tracking_predictions/jev.json').read_text())
    expected=[p for p in expected if lab.lookup[next(k for k,v in lab.lookup.items() if v['id']==p['image_id'])]['frame_id']<=241]
    actual=json.loads(Path(result['predictions']).read_text())
    if actual!=expected:raise AssertionError('native journal instrumentation changed B2 prefix predictions')
    fingerprint=lab.state_fingerprint(lab.prefix,lab.metadata)
    captures,state=lab.rebuild_memory_prefixes(selected)
    if fingerprint!=lab.state_fingerprint(state,lab.metadata):raise AssertionError('journal changed complete native state')
    if len(captures)!=3:raise AssertionError('journal did not recover selected MEMORY prefixes')
    value={'status':'PASS','through_frame':240,'native_B2_predictions_identical':True,
           'complete_mutable_state_and_metadata_identical':True,'captured_MEMORY_prefixes':len(captures),
           'journal_chunks':len(lab.journal_files),'future_counterfactual_actions_cached':False}
    save(OUT/'smoke/NATIVE_JOURNAL_TEST.json',value);print(json.dumps(value))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');main(p.parse_args().device)

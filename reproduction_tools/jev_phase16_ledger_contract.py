"""Actual earliest harmful prefix qualifies raw evidence provenance and restore."""
from jev_phase16_common import *
import torch
from jev_phase14_artifacts import load_dense
from gtr.modeling.jev_phase16.evidence_ledger import EvidenceLedger
from gtr.modeling.jev_native_state import fingerprint

def main():
    protect();torch.set_num_threads(1)
    source=read(XV/'train_commitment_prefixes_v1/video12/RESULT.json')
    entry=next(x for x in source['counterfactual_prefixes'] if x['key']==[12,3,1] and x['row']==6)
    descriptor=entry['prefix'];assert sha(descriptor['path'])==descriptor['SHA256']
    prefix=load_dense(descriptor['path'],map_location='cpu');ledger=EvidenceLedger.from_prefix(prefix)
    ledger.verify_raw_galleries(prefix['galleries'])
    state=ledger.state_dict();restored=EvidenceLedger();restored.load_state_dict(state)
    assert fingerprint(restored.state_dict())==fingerprint(state)
    first=next(iter(restored.records));before=fingerprint(ledger.state_dict())
    restored.records[first][0]['feature'].zero_()
    assert fingerprint(ledger.state_dict())==before,'snapshot aliases original feature storage'
    assert not any('GT' in k or 'gt'==k for rows in ledger.records.values() for r in rows for k in r)
    save(OUT/'ledger_contract/RESULT.json',dict(status='PASS',binding=binding(inputs=[descriptor],
        native_state=entry['starting_state_SHA256'],evaluator='exact real native Gallery tensor equality and deep restore',
        scope='TRAIN12 frame3 camera1 row6 actual original harmful prefix, not a scientific recovery outcome'),
        identities=len(ledger.records),observations=sum(len(r) for r in ledger.records.values()),
        raw_gallery_order_and_vectors_exact=True,snapshot_deepcopy=True,GT_fields_in_runtime_ledger=False))
    print('PHASE16_ACTUAL_GALLERY_LEDGER_CONTRACT_PASS',len(ledger.records),flush=True)

if __name__=='__main__':main()

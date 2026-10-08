"""Validated loader that exposes online evidence separately from offline labels."""
import gzip,json,math
from pathlib import Path

CANDIDATE_FIELDS=['GMT_score','score_minus_proposed','is_proposed','other_rows_proposing_candidate',
    'history_count','memory_count','last_seen_gap_frames','views_fraction','active','stale',
    'appearance_cosine','appearance_available']
ACTIONS=['ACCEPT_CURRENT','REASSOCIATE','START_NEW']

def finite(value):
    if isinstance(value,dict):return all(finite(v)for v in value.values())
    if isinstance(value,list):return all(finite(v)for v in value)
    return not isinstance(value,float)or math.isfinite(value)

def validate(record):
    online=record['online'];offline=record['offline_only']
    assert record['sequence']==(7 if record['role']=='train'else 6)
    assert record['question']=='MATCH_DECISION'and online['legal_actions']==ACTIONS
    assert online['factual_action']in ACTIONS and record['control_factual_state_parity']
    assert record['identical_prefix_RNG_and_exogenous_stream']and not record['GT_or_future_policy_inputs']
    assert record['coordinate_contract']=='cache0_annotation1'
    assert len(record['state_fingerprint'])==len(record['complete_state_fingerprint'])==len(record['snapshot_sha256'])==64
    assert len(online['state_features'])==64 and len(online['candidate_ids'])==len(set(online['candidate_ids']))>=2
    n=len(online['candidate_ids']);assert len(online['candidate_scores'])==len(online['candidate_evidence'])==n
    assert all(len(e)==len(CANDIDATE_FIELDS)for e in online['candidate_evidence'])
    assert all(len(r)==n for r in online['full_score_matrix'])
    assert online['candidate_scores']==online['full_score_matrix'][record['detection_index']]
    assert all(e[0]==score for e,score in zip(online['candidate_evidence'],online['candidate_scores']))
    assert finite(record)
    forbidden=('gt','ground_truth','future','utility','oracle','label')
    assert not any(any(s in k.lower()for s in forbidden)for k in online)
    probs=offline['target_probabilities'];assert set(probs)==set(ACTIONS)
    assert all(0<=p<=1 for p in probs.values())and abs(sum(probs.values())-1)<1e-6
    assert set(offline['utility_vector'])==set(ACTIONS)
    assert offline['oracle_utility']==max(offline['utility_vector'].values())
    assert set(offline['best_actions'])=={a for a,u in offline['utility_vector'].items()if abs(u-offline['oracle_utility'])<1e-8}
    assert set(offline['correct_candidate_ids'])<=set(online['candidate_ids'])
    assert offline['weight']in (0.,1.)and offline['weight']==float(offline['informative'])
    assert not offline['unknown_target']or offline['weight']==0.
    for action in ACTIONS:
        assert set(record['branches'][action]['fingerprints'])==set(record['branches']['CONTROL']['fingerprints'])
    if 'data_gate'in record:
        assert record['offline_only']['effective_training_weight']==float(record['data_gate']['record_eligible'])
        assert len(record['data_gate']['original_source_label_sha256'])==64
        if record['data_gate']['quarantine_reason']is not None:assert record['offline_only']['effective_training_weight']==0.
    return record

class NativeCausalMiniSet:
    def __init__(self,path,role=None):
        path=Path(path);opener=gzip.open if path.suffix=='.gz'else open
        with opener(path,'rt')as stream:self.records=[validate(json.loads(line))for line in stream if line.strip()]
        keys=[r['event_id']for r in self.records];assert len(keys)==len(set(keys))
        if role is not None:assert all(r['role']==role for r in self.records)
    def __len__(self):return len(self.records)
    def __getitem__(self,index):return self.records[index]
    def inference_input(self,index):
        # Semantic IDs/order accompany tensors for masking/assignment; IDs are not scalar features.
        return self.records[index]['online']

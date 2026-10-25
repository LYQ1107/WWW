from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from worldobject_synthetic import core,obs
from test_worldobject_memory import evidence
from audit_worldobject_v3_frozen_birth_counterfactual import FrozenBirthCounterfactual,TINY_BENEFIT


def test_same_real_history_shadows_expose_positive_wrong_birth_without_changing_actual_IDs():
    tracker=core();tracker.step([obs()],0)
    observer=FrozenBirthCounterfactual(tracker)
    items=[obs(timestamp=1),obs(2,timestamp=1,feature=(0,1,0))]
    learned=evidence(1,2);learned['assignment_logits'][:]=[-10,-10];learned['assignment_logits'][0,0]=8
    learned['null_logits'][:]=-2;learned['birth_logits'][:]=-60
    learned['birth_pair_logits']=np.full((2,2),10.);learned['birth_pair_mask']=np.triu(np.ones((2,2),bool),1)
    observer.start(items,['A','B'],1,{1:'A'},1)
    tracker.step(items,1,evidence=learned);observer.finish()
    assert tracker.bank.last_assignments[(1,0)]==1
    assert tracker.bank.last_assignments[(2,0)]!=1
    assert observer.counts['actual']['selected_wrong_joint_birth']==0
    assert observer.counts['no_history_tiny_positive']['selected_wrong_joint_birth']==1
    assert 0<TINY_BENEFIT<1e-30


def test_shadow_groups_never_write_hypothetical_global_ids_into_actual_bank():
    tracker=core();observer=FrozenBirthCounterfactual(tracker)
    items=[obs(),obs(2,feature=(0,1,0))];learned=evidence(0,2)
    learned['birth_pair_logits']=np.full((2,2),-30.);learned['birth_pair_mask']=np.triu(np.ones((2,2),bool),1)
    observer.start(items,['A','B'],0,{},1);tracker.step(items,0,evidence=learned);observer.finish()
    assert tracker.bank.next_id==3 and set(tracker.bank.objects)=={1,2}
    assert tracker.bank.last_assignments[(1,0)]!=tracker.bank.last_assignments[(2,0)]

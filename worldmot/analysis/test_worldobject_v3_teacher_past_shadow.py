from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from worldobject_synthetic import core,obs
from audit_worldobject_v3_teacher_past_shadow import shadow_prediction


def test_shadow_uses_ordinary_current_decision_without_mutating_original_history():
    tracker=core();tracker.step([obs()],0)
    before=tracker.bank.objects[1].identity_embedding.copy()
    assignments=shadow_prediction(tracker,[obs(feature=(0,1,0),timestamp=1)],1)
    assert assignments[(1,0)]!=1
    assert tracker.bank.next_id==2 and tracker.bank.last_timestamp==0
    assert tracker.bank.last_assignments[(1,0)]==1
    assert (tracker.bank.objects[1].identity_embedding==before).all()


def test_shadow_NULL_does_not_increment_original_missed_or_delete_objects():
    tracker=core();tracker.step([obs()],0)
    assert shadow_prediction(tracker,[],1)=={}
    assert tracker.bank.objects[1].missed==0
    assert tracker.bank.objects[1].status!='DELETED'

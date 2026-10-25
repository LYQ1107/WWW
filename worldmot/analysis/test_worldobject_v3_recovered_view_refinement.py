from refine_worldobject_v3_recovered_view_events import classify_next_view_event


def test_preexisting_other_camera_split_is_not_a_recovered_view_break():
    row=classify_next_view_event({2},169,{1:1,2:169},set(),set())
    assert row['next_recovered_view_kept_old_ID']
    assert not row['next_recovered_view_ID_change']
    assert not row['next_new_ID_on_recovered_view']


def test_actual_recovered_camera_new_birth_and_geometry_veto_are_counted():
    row=classify_next_view_event({2},158,{1:1,2:162},{2},{2})
    assert row['next_recovered_view_ID_change']
    assert row['next_new_ID_on_recovered_view']
    assert row['next_recovered_view_geometry_rejected']


def test_new_id_only_in_the_other_camera_is_not_recovery_chatter():
    row=classify_next_view_event({2},169,{1:999,2:169},{1},set())
    assert row['other_view_new_ID_only']
    assert not row['next_new_ID_on_recovered_view']


def test_no_current_clean_observation_in_recovered_view_is_separate():
    row=classify_next_view_event({2},169,{1:1},set(),set())
    assert not row['next_recovered_view_observed']
    assert not row['next_recovered_view_ID_change']

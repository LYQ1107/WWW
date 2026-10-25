from recover_worldobject_v3_launch_guard import boolean_slot_count


def test_cpu_none_values_do_not_break_gpu_counting():
    assert boolean_slot_count([None,False,True,None,True])==2


def test_boolean_only_legacy_case_is_unchanged():
    values=[True,False,False,True]
    assert boolean_slot_count(values)==sum(values)


def test_six_device_limit_is_preserved_with_cpu_jobs():
    assert boolean_slot_count([None]*4+[True]*6)==6

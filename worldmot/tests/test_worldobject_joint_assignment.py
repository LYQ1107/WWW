from worldobject_synthetic import obs,core,gid


def test_close_people_with_distinct_identity_are_not_merged():
    c=core();r=c.step([obs(1,0),obs(2,0,feature=(0,1,0))],0)
    assert gid(r,1)!=gid(r,2)


def test_similar_appearance_spatially_separated_keeps_two_ids():
    c=core();r=c.step([obs(1,0),obs(2,0,center=(20,0,3))],0)
    assert gid(r,1)!=gid(r,2)


def test_history_jointly_explains_two_cameras_without_premerge():
    c=core();first=c.step([obs(1)],0);id1=gid(first,1)
    r=c.step([obs(2,timestamp=1),obs(1,timestamp=1)],1)
    assert gid(r,1)==gid(r,2)==id1
    assert r["bank"]["premerged_pairs"] is False


def test_occlusion_outputs_only_current_detection():
    c=core();first=c.step([obs(1),obs(2)],0);id1=gid(first,1)
    r=c.step([obs(2,timestamp=1)],1)
    assert "1" not in r["tracks_2d_by_camera"] and gid(r,2)==id1
    empty=c.step([],2)
    assert empty["tracks_2d_by_camera"]=={} and c.bank.objects[id1].status=="LOST"


def test_conflicting_pair_competes_with_consistent_historical_option():
    c=core();old=c.step([obs(1)],0);identity=gid(old,1)
    r=c.step([obs(1,timestamp=1),obs(2,0,center=(20,0,3),timestamp=1),obs(2,1,timestamp=1)],1)
    assert gid(r,1)==gid(r,2,1)==identity
    assert gid(r,2,0)!=identity


def test_one_detection_per_camera_per_id_and_never_reuse_deleted_id():
    c=core(max_missed=1)
    r=c.step([obs(1,0),obs(1,1)],0)
    ids={x["global_id"] for x in r["tracks_2d_by_camera"]["1"]}
    assert len(ids)==2
    c.step([],1);c.step([],2)
    r=c.step([obs(1,timestamp=3)],3)
    assert gid(r,1)>max(ids)


def test_large_component_is_deterministic_and_bounded():
    candidates=[obs(1,i,center=(i*.001,0,3)) for i in range(60)]+[obs(2,i,center=(i*.001,0,3)) for i in range(60)]
    c=core(beam_width=8,max_exact_states=16,top_k=2)
    a=c.step(candidates,0)
    other=core(beam_width=8,max_exact_states=16,top_k=2)
    b=other.step(list(reversed(candidates)),0)
    assert a["tracks_2d_by_camera"]==b["tracks_2d_by_camera"]
    assert all(x["state_width_limit"]<=16 for x in a["bank"]["components"])
    assert a["bank"]["expanded_states"]<10000
    assert a["bank"]["all_detections_assigned"]


def test_capacity_limit_keeps_every_current_box_as_tentative_birth():
    c=core(max_observations=2)
    r=c.step([obs(1,i) for i in range(3)],0)
    assert len(r["tracks_2d_by_camera"]["1"])==3
    assert r["bank"]["fallback"]=="SEARCH_LIMIT_SINGLETON_TENTATIVE_BIRTHS"

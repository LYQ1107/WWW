import json
from pathlib import Path
import pytest


def test_export_preserves_all_camera_streams_without_opening_GT(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from export_fullvideo_predictions import export
    metadata=tmp_path/'metadata.json';pred=tmp_path/'pred/scene_01';pred.mkdir(parents=True)
    metadata.write_text(json.dumps({'videos':[{'id':1,'file_name':'sample'}],
        'images':[{'video_id':1,'frame_id':t,'view_id':v} for t in range(1,4) for v in [1,2]]}))
    (pred/'manifest.json').write_text(json.dumps({'status':'COMPLETE','full_video':True}))
    rows=[{'timestamp':t,'tracks_2d_by_camera':{str(v):[{'global_id':7,'bbox_xywh':[1,2,3,4]}] for v in [1,2]}}
          for t in range(1,4)]
    (pred/'tracks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=export(metadata,pred.parent,tmp_path/'submission')
    assert report['GT_read'] is False and report['timestamps']==3
    assert len((tmp_path/'submission/sample_View1.txt').read_text().splitlines())==3
    assert len((tmp_path/'submission/sample_View2.txt').read_text().splitlines())==3
    (pred/'tracks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:2]))
    with pytest.raises(RuntimeError,match='incomplete timestamp'):
        export(metadata,pred.parent,tmp_path/'bad_submission')


def test_interrupted_stream_is_archived_and_complete_mismatch_is_rejected(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from preserve_inference_attempts import prepare_scene
    scene=tmp_path/'scene_01';scene.mkdir();(scene/'tracks.jsonl').write_text('negative attempt\n')
    (scene/'manifest.json').write_text(json.dumps({'status':'RUNNING'}))
    assert prepare_scene(scene,{'checkpoint_sha256':'first'})
    archived=list((tmp_path/'.interrupted_attempts').glob('*/tracks.jsonl'))
    assert len(archived)==1 and archived[0].read_text()=='negative attempt\n'
    (scene/'manifest.json').write_text(json.dumps({'status':'COMPLETE','checkpoint_sha256':'first'}))
    assert prepare_scene(scene,{'checkpoint_sha256':'first'}) is False
    with pytest.raises(RuntimeError,match='different checkpoint'):
        prepare_scene(scene,{'checkpoint_sha256':'second'})


def test_official_format_conversion_does_not_round_groundtruth_coordinates(tmp_path):
    from worldmot.evaluation.official_fullvideo import read_mot,write_mot
    rows=[[1,4,.1234567891234567,21.987654321234567,32.333333333333336,43.77777777777778,1,1,1,-1]]
    path=tmp_path/'gt.txt';write_mot(path,rows)
    assert read_mot(path)==rows


def test_posthoc_identity_diagnostics_keep_official_IDSW_separate(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from analyze_fulltrain_test_failures import analyze
    from worldmot.evaluation.official_fullvideo import sha
    root=tmp_path/'experiment';(root/'audit').mkdir(parents=True)
    groundtruth=[]
    for view in [1,2]:
        path=root/f'GT_View{view}.txt';path.write_text('1,7,10,20,30,40,1,1,1\n2,7,10,20,30,40,1,1,1\n')
        groundtruth.append({'view':view,'path':str(path),'sha256':sha(path)})
    (root/'audit/data_audit.json').write_text(json.dumps({'test':{'scenes':[{'scene':1,'official_gt_legal':True,'raw_gt':groundtruth}]}}))
    (root/'audit/test_metadata.json').write_text(json.dumps({'images':[]}))
    pred=root/'test/synthetic/scene_01';pred.mkdir(parents=True)
    (pred/'manifest.json').write_text(json.dumps({'status':'COMPLETE','full_video':True}))
    rows=[]
    for t in [1,2]:
        boxes={str(v):[{'global_id':11 if t==2 and v==2 else 10,'bbox_xywh':[10,20,30,40]}] for v in [1,2]}
        rows.append({'timestamp':t,'tracks_2d_by_camera':boxes,'detections_by_camera':boxes})
    (pred/'tracks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    ev=root/'test_evaluation/synthetic';ev.mkdir(parents=True)
    (ev/'metrics.json').write_text(json.dumps({'status':'COMPLETE_VALID_SCOPE','TrackEval':{'combined':{'IDSW':99}}}))
    target=tmp_path/'results/failures.json';analyze(root,target)
    result=json.loads(target.read_text());method=result['methods']['synthetic']
    assert method['official_IDSW']==99
    assert method['counts']['PERSISTENT_ID_CHANGE_DIAGNOSTIC']==1
    assert method['counts']['SAME_PERSON_SPLIT_ACROSS_CAMERAS']==1
    assert result['matching_is_not_TrackEval_or_official_IDSW'] is True
    assert all(sha(r['path'])==r['sha256'] for r in groundtruth)


def test_empty_experiment_cannot_publish_complete_test_scores(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from collect_fulltrain_results import collect
    root=tmp_path/'experiment';root.mkdir();out=tmp_path/'results'
    result=collect(root,out)
    assert result['complete'] is False and result['official_methods']==0
    assert not (out/'worldmot_v2_test_official_metrics.json').exists()
    with pytest.raises(RuntimeError,match='final artifacts incomplete'):
        collect(root,out,require_complete=True)


def test_image_preflight_checks_actual_pixels_and_keeps_failure_evidence(tmp_path,monkeypatch):
    import hashlib
    from PIL import Image
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_test_rgb import check_image
    path=tmp_path/'frame.jpg';Image.new('RGB',(16,8),(80,90,100)).save(path)
    item={'id':1,'video_id':1,'view_id':2,'frame_id':3,'file_name':'frame.jpg','width':16,'height':8}
    valid=check_image(item,tmp_path)
    assert valid['status']=='RGB_DECODE_OK'
    assert valid['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    mismatched=check_image({**item,'width':17},tmp_path)
    assert mismatched['status']=='RGB_DECODE_FAILED' and 'dimensions' in mismatched['error']
    path.write_bytes(b'corrupt image')
    corrupted=check_image(item,tmp_path)
    assert corrupted['status']=='RGB_DECODE_FAILED' and corrupted['image_id']==1


def test_test_queue_reuses_free_gpu_without_overlapping_jobs(tmp_path,monkeypatch):
    import collections
    import threading
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import orchestrate_fulltrain_test as driver
    monkeypatch.setattr(driver,'GPUS',[0,2])
    active=collections.Counter();completed=[];lock=threading.Lock()
    first_started=threading.Event();release_first=threading.Event()
    def job(index,gpu):
        with lock:
            active[gpu]+=1
            assert active[gpu]==1
        try:
            if index==0:
                first_started.set()
                assert release_first.wait(5)
            elif index==1:
                assert first_started.wait(5)
            elif index==2:
                assert gpu==2  # GPU 0 is still occupied by the first job.
                release_first.set()
            with lock:completed.append(index)
            return index,gpu
        finally:
            with lock:active[gpu]-=1
    experiment=driver.Experiment(tmp_path)
    result=experiment.parallel([lambda gpu,i=i:job(i,gpu) for i in range(5)],dynamic=True)
    assert sorted(completed)==list(range(5))
    assert sorted(i for i,gpu in result)==list(range(5))
    assert all(count==0 for count in active.values())

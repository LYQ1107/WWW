"""Real GTR get_asso/GTA and CausalGMT, with stubbed perception/transformer."""
import tempfile,types,unittest,cv2,numpy as np,torch,sys,random
from pathlib import Path
from common import ROOT,RUNTIME
from detectron2.config import get_cfg
from gtr.config import add_gtr_config
from gtr.geometry.runtime import GeometryRuntime
from gtr.geometry.online_runner import CausalGMT
from gtr.geometry.sparse_scheduler import SparseScheduler
from gtr.geometry.update_signals import identity_update_signal,CausalBackgroundMotion
from test_empty_tracking import detections,model_stub

def configured_runtime(policy='joint'):
    cfg=get_cfg();add_gtr_config(cfg);geo=cfg.MODEL.GEO
    geo.ENABLED=True;geo.FEATURE_ENABLED=True;geo.FEATURE_WEIGHT=.5
    geo.FEATURE_BACKBONE='gmt';geo.FEATURE_LAYERS=[-1];geo.PROJECTION_DIM=0
    geo.POOLING='reid';geo.SYMMETRIZE_VIEW_ORDER=False;geo.UPDATE_INTERVAL=3
    geo.CALIBRATION_PATH=str(ROOT/'manifests/calibration_gmt_reid_absolute_train_v1.json')
    geo.SPARSE_UPDATE_ENABLED=True;geo.UPDATE_POLICY=policy
    geo.COMPUTE_BUDGET=3;geo.BUDGET_WINDOW=10;geo.MIN_UPDATE_INTERVAL=2
    return GeometryRuntime(geo)

class Tests(unittest.TestCase):
    def test_causal_tracker_prefix_budget_and_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'frame.png';rng=np.random.default_rng(42)
            cv2.imwrite(str(path),rng.integers(0,256,(180,320),dtype=np.uint8))
            def run(length):
                model=model_stub();model.geo_runtime=configured_runtime()
                def inference(batch,*args,**kwargs):
                    inst=detections(2);inst.reid_features=torch.eye(2,8)
                    return [inst]
                def transformer(instances,reid,*args):
                    k=args[1];query=instances[k]
                    refs=torch.cat([x.reid_features for i,x in enumerate(instances) if i!=k])
                    logits=2*query.reid_features@refs.T
                    if int(query.geo_frame[0])%7==5:logits.zero_()
                    boxes,_=model.roi_heads._get_boxes_time(instances)
                    return [logits],boxes,None,None,None,None
                model.inference=inference;model.roi_heads._forward_transformer=transformer
                tracker=CausalGMT(model);output=[]
                before=torch.get_rng_state().clone()
                for timestamp in range(1,length+1):
                    records=[dict(frame_id=timestamp,view_id=v,file_name=str(path),height=180,width=320) for v in [1,2]]
                    result=tracker.step(records)
                    output.append([r['instances'].track_ids.tolist() for r in result])
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                return output,model.geo_runtime
            short,a=run(25);long,b=run(35)
            self.assertEqual(short,long[:25]);self.assertEqual(a.schedule_trace,b.schedule_trace[:25])
            chosen=[r['chosen'] for r in a.schedule_trace]
            self.assertEqual(a.calls,sum(chosen));self.assertGreater(a.calls,0)
            for t,row in enumerate(a.schedule_trace):
                self.assertLessEqual(sum(chosen[max(0,t-9):t+1]),3)
                self.assertTrue(row['identity_source_frame'] is None or row['identity_source_frame']<t)
            self.assertEqual(a.vggt_calls+a.dinov2_calls,0)

    def test_actual_gta_signal_splits_history_frames(self):
        model=model_stub();refs=[detections(1),detections(1)]
        for i,ref in enumerate(refs):
            ref.track_ids=torch.tensor([float(i+1)]);ref.geo_camera=torch.tensor([1])
        query=detections(1);query.geo_camera=torch.tensor([2])
        margin,handoff=identity_update_signal(torch.tensor([[3.,-3.]]),refs+[query],2,model.roi_heads._activate_asso)
        expected=float((torch.sigmoid(torch.tensor(3.))-torch.sigmoid(torch.tensor(-3.)))/torch.sigmoid(torch.tensor(3.)))
        self.assertAlmostEqual(margin,expected,places=6);self.assertTrue(handoff)

    def test_no_motion_on_static_or_textureless_background(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'frame.png';cv2.imwrite(str(path),np.zeros((180,320),np.uint8))
            signal=CausalBackgroundMotion();records=[dict(view_id=1,file_name=str(path))];boxes=[torch.empty(0,4)]
            self.assertEqual(signal.observe(records,boxes)[0],0.)
            self.assertEqual(signal.observe(records,boxes)[0],0.)

    def test_current_background_translation_signal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'frame.png';rng=np.random.default_rng(42)
            image=cv2.GaussianBlur(rng.integers(0,256,(180,320),dtype=np.uint8),(5,5),0)
            cv2.imwrite(str(path),image);signal=CausalBackgroundMotion()
            records=[dict(view_id=1,file_name=str(path))];boxes=[torch.empty(0,4)]
            signal.observe(records,boxes)
            moved=cv2.warpAffine(image,np.float32([[1,0,5],[0,1,0]]),(320,180))
            cv2.imwrite(str(path),moved);motion,details=signal.observe(records,boxes)
            self.assertTrue(details[0]['valid'])
            self.assertAlmostEqual(motion,5/np.hypot(320,180),delta=.002)

    def test_budget_zero_and_nonmonotonic_even_skipped(self):
        scheduler=SparseScheduler(policy='every',budget=0)
        self.assertFalse(scheduler.should_update(1))
        with self.assertRaises(ValueError):scheduler.should_update(1)

    def test_symmetric_encoder_cost_and_integrated_failure_fallback(self):
        cfg=get_cfg();add_gtr_config(cfg);geo=cfg.MODEL.GEO
        geo.ENABLED=True;geo.FEATURE_ENABLED=True;geo.FEATURE_WEIGHT=.5
        geo.FEATURE_LAYERS=[17];geo.PROJECTION_DIM=0
        geo.CALIBRATION_PATH=str(ROOT/'manifests/calibration_native_l17_absolute_train_v1.json')
        geo.SPARSE_UPDATE_ENABLED=True;geo.UPDATE_POLICY='every'
        geo.COMPUTE_BUDGET=3;geo.BUDGET_WINDOW=10;geo.MIN_UPDATE_INTERVAL=1
        runtime=GeometryRuntime(geo)
        def fail(*args,**kwargs):
            random.random();np.random.random();torch.rand(1)
            raise RuntimeError('simulated encoder OOM')
        runtime.vggt=types.SimpleNamespace(extract=fail)
        sys.path.insert(0,str(RUNTIME/'sources/vggt'))
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'frame.png';cv2.imwrite(str(path),np.zeros((180,320),np.uint8))
            for frame in range(20):
                refs,query=detections(1),detections(1);refs.track_ids=torch.tensor([1.])
                records=[dict(frame_id=frame+1,view_id=v,file_name=str(path),height=180,width=320) for v in [1,2]]
                py_before=random.getstate();np_before=np.random.get_state();torch_before=torch.get_rng_state().clone()
                runtime.observe([refs,query],records,frame)
                self.assertEqual(py_before,random.getstate())
                self.assertTrue(np.array_equal(np_before[1],np.random.get_state()[1]))
                self.assertEqual(np_before[2:],np.random.get_state()[2:])
                self.assertTrue(torch.equal(torch_before,torch.get_rng_state()))
                base=[torch.ones(1,1)]
                self.assertTrue(torch.equal(runtime.adapt(base,[refs,query],1)[0],base[0]))
                self.assertFalse(query.geo_valid.any())
        self.assertEqual(runtime.update_cost,2)
        self.assertEqual(runtime.failed,2)
        self.assertEqual(runtime.vggt_calls,2)
        self.assertTrue(all(row['reserved_units_in_window']<=3 for row in runtime.schedule_trace))

if __name__=='__main__':unittest.main()

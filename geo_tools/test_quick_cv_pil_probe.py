"""CPU-only tests for quick_cv_pil_probe; synthetic data are NOT VisionTrack evidence."""
import hashlib
import json
import tempfile
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from quick_cv_pil_probe import (LowRankPredictor, build_episodes, read_cache,
                                run_screen, tensor_episode)

class TestFastProbe(unittest.TestCase):
    def make_cache(self, root, video):
        folder = Path(root)/'history_train'/f'history_train_keyframe_v1_v{video}'
        folder.mkdir(parents=True)
        rng = np.random.RandomState(video)
        metadata=[];labels=[];gmt=[];dino=[];vggt=[]
        for frame in range(1,61,3):
            for camera in [1,2]:
                for identity in [7,8,9]:
                    arr=np.zeros(1152,dtype=np.float32)
                    arr[(identity-7)*8+camera]=1.5
                    arr[(identity-7)*8]=5.0
                    arr += .008*rng.randn(1152).astype(np.float32)
                    metadata.append(dict(frame=frame,camera=camera))
                    labels.append(identity)
                    gmt.append(arr.astype(np.float16))
                    dino.append(arr[:256].astype(np.float16))
                    vggt.append(np.tile(arr[:256],16).astype(np.float16))
        np.savez_compressed(folder/'targets.npz',labels=np.array(labels),gmt=np.array(gmt),dinov2=np.array(dino),vggt=np.array(vggt))
        (folder/'observations.json').write_text(json.dumps(metadata))
        sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
        (folder/'manifest.json').write_text(json.dumps(dict(
            video_id=video,source_split='association_train',neutral_prefix40_exact=True,
            encoder_failures=0,fit_video_ids=[5,8],validation_video_ids=[12],
            artifacts={p:sha(folder/p) for p in ['targets.npz','observations.json']})))
        return folder

    def test_end_to_end_four_arms_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            for video in [5,8,12]:self.make_cache(d,video)
            out=Path(d)/'out'
            result=run_screen(Path(d),out,steps=8,device='cpu')
            self.assertEqual(result['status'],'COMPLETE_OFFLINE_SCREEN')
            self.assertEqual(set(result['arms']),{'linear','contrastive','predictive','predictive_no_camera'})
            self.assertEqual(result['deployment_gate'],'NO_PREDICTIVE_SPECIFIC_SCREENING_GAIN')
            self.assertEqual(len(list(out.glob('*.pth'))),4)
            self.assertTrue((out/'screen.json').exists())
            cache=read_cache(Path(d),12)
            ep=build_episodes(cache)
            self.assertTrue(ep)
            for e in ep:
                for rows in e['candidate_rows']:
                    for i in rows:
                        self.assertLess(cache['meta'][i]['frame'],e['frame'])
                        self.assertNotEqual(cache['meta'][i]['camera'],e['camera'])
                q,h,age=tensor_episode(cache,e,torch.device('cpu'))
                self.assertEqual(q.numel(),1024)
                self.assertTrue((age>0).all())

    def test_tampered_cache_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=self.make_cache(d,5)
            (folder/'observations.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                read_cache(Path(d),5)

    def test_zero_initialization_exact_identity_and_camera_ablation(self):
        torch.manual_seed(0)
        h=torch.nn.functional.normalize(torch.randn(3,1024),dim=-1)
        age=torch.tensor([.1,.2,.3])
        for arm in [(True,True),(False,True),(True,False)]:
            m=LowRankPredictor(nonlinear=arm[0],use_camera=arm[1])
            self.assertTrue(torch.allclose(m(h,1,age),h,atol=1e-7))

if __name__=='__main__':unittest.main()

"""Synthetic CPU contracts for past-only native GTA CV-PIL screen. No real-data gain."""
import hashlib
import json
import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))

import test_quick_cv_pil_probe as fixtures
from quick_cv_pil_native import load_native, native_episodes, fit_model, report


class TestNativeScreen(unittest.TestCase):
    def make_native(self,d,video):
        base=fixtures.TestFastProbe().make_cache(d,video)
        from quick_cv_pil_probe import read_cache
        item=read_cache(Path(d),video)
        observations=item['meta']
        events=[]
        for t in sorted({m['frame'] for m in observations}):
            q=[i for i,m in enumerate(observations) if m['frame']==t and m['camera']==2]
            ref=[i for i,m in enumerate(observations) if m['frame']==t-3 and m['camera']==1]
            if q and ref:
                logits=torch.randn(len(q),len(ref))
                events.append(dict(query=q,reference=ref,logits=logits))
        torch.save(events,base/'events.pt')
        manifest=json.loads((base/'manifest.json').read_text())
        manifest['artifacts']['events.pt']=hashlib.sha256((base/'events.pt').read_bytes()).hexdigest()
        (base/'manifest.json').write_text(json.dumps(manifest))
        return base

    def test_actual_candidate_protocol_and_training(self):
        with tempfile.TemporaryDirectory() as d:
            data={}
            episodes={}
            for v in [5,8,12]:
                self.make_native(d,v)
                data[v]=load_native(Path(d),v)
                episodes[v]=native_episodes(data[v])
                self.assertGreaterEqual(len(episodes[v]),8)
                for e in episodes[v]:
                    self.assertEqual(sum(data[v]['labels'][r[0]]==e['target_id'] for r in e['candidate_rows']),1)
                    for rows in e['candidate_rows']:
                        self.assertLess(data[v]['meta'][rows[0]]['frame'],e['frame'])
                        self.assertNotEqual(data[v]['meta'][rows[0]]['camera'],e['camera'])
            device=torch.device('cpu')
            b=report(data[12],episodes[12],device,mode='native')
            self.assertEqual(b['clauses'],len(episodes[12]))
            fit=episodes[5]+episodes[8]
            for arm in ['linear','contrastive','predictive','predictive_no_camera']:
                model,logs=fit_model(arm,data,fit,device,steps=3)
                r=report(data[12],episodes[12],device,model=model)
                self.assertTrue(0<=r['recall1']<=1)
                self.assertEqual(logs[0]['step'],1)

    def test_tampered_event_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=self.make_native(d,5)
            (folder/'events.pt').write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError,'Native event SHA mismatch'):
                load_native(Path(d),5)


if __name__=='__main__':
    unittest.main()

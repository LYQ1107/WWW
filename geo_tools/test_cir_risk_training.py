"""Masking, dependence weighting and conservative threshold calibration."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cir_risk_training import read_risk_training,calibrate_risk_threshold


class RiskTrainingContracts(unittest.TestCase):
    def fixture(self,records):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);path=Path(temp.name)/'rows.jsonl.gz'
        with gzip.open(path,'wt') as stream:
            for record in records:stream.write(json.dumps(record)+'\n')
        return path,hashlib.sha256(path.read_bytes()).hexdigest()
    def row(self,image,label,person=1,scene=1):
        return dict(key=[scene,image,0],features=[0.]*12,risk_label=label,person=person,
            source_state='Original GMT history + same-context C1 primary joint proposal, no mutated future/veto utility yet')
    def test_UNKNOWN_mask_and_person_not_in_features(self):
        rows=[self.row(1,0),self.row(2,1),self.row(3,None,None)]
        path,digest=self.fixture(rows);data=read_risk_training(path,digest,{1})
        self.assertEqual(data['labels'].tolist(),[0.,1.]);self.assertEqual(data['stats']['UNKNOWN_queries'],1)
        rows[-1]['person']=99;path,digest=self.fixture(rows);other=read_risk_training(path,digest,{1})
        self.assertTrue(np.array_equal(data['all_features'].numpy(),other['all_features'].numpy()))
    def test_duplicate_or_foreign_sources_and_wrong_SHA_rejected(self):
        path,digest=self.fixture([self.row(1,0),self.row(1,1)])
        with self.assertRaises(ValueError):read_risk_training(path,digest,{1})
        path,digest=self.fixture([self.row(1,0),self.row(2,1,scene=2)])
        with self.assertRaises(ValueError):read_risk_training(path,digest,{1})
        with self.assertRaises(ValueError):read_risk_training(path,'wrong',{1,2})
    def test_repeated_person_is_one_dependence_group_and_classes_balance(self):
        path,digest=self.fixture([self.row(1,0),self.row(2,1),self.row(3,1),self.row(4,1)])
        data=read_risk_training(path,digest,{1});w=data['weights'].numpy();y=data['labels'].numpy()
        self.assertEqual(data['stats']['wrong_scene_person_groups'],1)
        self.assertAlmostEqual(float(w[y==0].sum()),float(w[y==1].sum()),places=5)
        self.assertAlmostEqual(float(w.mean()),1.,places=5)
    def test_tied_score_cannot_secretly_choose_positive_record(self):
        d=calibrate_risk_threshold(np.array([.9,.9,.8,.1]),np.array([1,0,1,0]),0.)
        self.assertEqual(d['TP'],0);self.assertEqual(d['FP'],0)
    def test_strict_threshold_calibration_matches_deployment(self):
        d=calibrate_risk_threshold(np.array([.9,.8,.7,.1]),np.array([1,1,0,0]),0.)
        self.assertEqual(d['threshold'],.7);self.assertEqual(d['TP'],2);self.assertEqual(d['FP'],0)
    def test_equal_F1_prefers_more_conservative_threshold(self):
        d=calibrate_risk_threshold(np.array([.9,.8,.7,.6]),np.array([1,0,0,1]),1.)
        self.assertEqual(d['threshold'],.8);self.assertEqual(d['TP'],1);self.assertEqual(d['FP'],0)


if __name__=='__main__':unittest.main()

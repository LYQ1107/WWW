"""Actual-data integrity and adverse-record tests for the portable native loader."""
import copy,json,unittest
from jev_phase7_common import *
from jev_phase7_miniset import NativeCausalMiniSet,validate

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train=NativeCausalMiniSet(REPORTS/'miniset_v1/train.jsonl.gz','train')
        cls.val=NativeCausalMiniSet(REPORTS/'miniset_v1/validation.jsonl.gz','validation')
    def test_real_distinct_splits_and_no_GT_in_inference(self):
        self.assertEqual(len(self.train),16);self.assertEqual(len(self.val),16)
        self.assertFalse({r['event_id']for r in self.train.records}&{r['event_id']for r in self.val.records})
        for data in (self.train,self.val):
            for index in range(len(data)):
                inputs=data.inference_input(index)
                self.assertNotIn('offline_only',inputs);self.assertNotIn('GT',inputs);self.assertNotIn('utility_vector',inputs)
                factual=data[index]['online']['factual_action']
                self.assertEqual(data[index]['branches']['CONTROL']['fingerprints'],data[index]['branches'][factual]['fingerprints'])
    def test_reject_wrong_role_candidate_order_and_future_field(self):
        r=copy.deepcopy(self.train[1]);r['sequence']=6
        with self.assertRaises(AssertionError):validate(r)
        r=copy.deepcopy(self.train[1]);r['online']['candidate_scores']=list(reversed(r['online']['candidate_scores']))
        with self.assertRaises(AssertionError):validate(r)
        r=copy.deepcopy(self.train[1]);r['online']['future_GT']=1
        with self.assertRaises(AssertionError):validate(r)
    def test_reject_nonfinite_probabilities_and_oracle(self):
        r=copy.deepcopy(self.train[1]);r['online']['state_features'][0]=float('nan')
        with self.assertRaises(AssertionError):validate(r)
        r=copy.deepcopy(self.train[1]);r['offline_only']['target_probabilities']['START_NEW']=1.2
        with self.assertRaises(AssertionError):validate(r)
        r=copy.deepcopy(self.train[1]);r['offline_only']['oracle_utility']+=1
        with self.assertRaises(AssertionError):validate(r)
    def test_unanchored_prefix_is_quarantined(self):
        records=self.train.records+self.val.records
        uncertain=[r for r in records if r['data_gate']['quarantine_reason']=='UNANCHORED_PREFIX_IDENTITY']
        self.assertGreater(len(uncertain),0)
        self.assertTrue(all(r['offline_only']['effective_training_weight']==0 for r in uncertain))
        self.assertTrue(all(not r['data_gate']['global_fitting_gate_passed']for r in records))

if __name__=='__main__':
    result=unittest.main(exit=False)
    save(REPORTS/'MINISET_TESTS.json',{'status':'PASS'if result.result.wasSuccessful()else 'FAIL','tests':result.result.testsRun,
                                    'failures':len(result.result.failures),'errors':len(result.result.errors),'binding':binding()})
    if not result.result.wasSuccessful():raise SystemExit(1)

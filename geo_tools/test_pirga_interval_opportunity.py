import unittest
import numpy as np
from pirga_interval_opportunity import interval_opportunity


class IntervalTests(unittest.TestCase):
    def test_strict_threshold_and_complete_joint_competition(self):
        a=np.array([[.8,.7],[.79,.1]])
        impossible=interval_opportunity(a,np.zeros_like(a),[.1,.1],0,[0])
        self.assertTrue(impossible['certified_impossible'])
        possible=interval_opportunity(a,np.full_like(a,.3),[.1,.1],0,[0])
        self.assertFalse(possible['certified_impossible']);self.assertTrue(possible['interval_witness_found'])
        threshold=interval_opportunity(np.array([[.1]]),np.array([[0.]]),[.1],0,[0])
        self.assertTrue(threshold['certified_impossible'])

    def test_rectangular_assignment_and_multi_positive_aliases(self):
        a=np.array([[.7,.6],[.8,.1],[.3,.9]])
        result=interval_opportunity(a,np.full_like(a,.2),[.1,.1],0,[0,1])
        self.assertTrue(result['interval_witness_found']);self.assertFalse(result['certified_impossible'])
        with self.assertRaises(ValueError):interval_opportunity(a,-np.ones_like(a),[.1,.1],0,[0])

if __name__=='__main__':unittest.main()

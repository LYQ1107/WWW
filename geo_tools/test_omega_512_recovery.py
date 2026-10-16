"""Check admission cannot combine incompatible orders or scenes; preserve matcher defaults."""
import unittest
from unittest.mock import patch
import numpy as np
from omega_512_background_recovery import joint_support
from omega_modules import geometry_module

class Tests(unittest.TestCase):
    def test_joint_scene_admission(self):
        rows=[]
        for pair,scene in [('a',4),('b',4),('c',8)]:
            for order in ['AB','BA']:
                rows.append(dict(arm='C1',method='omega512',pair_id=pair,scene=scene,order=order,valid=True))
        self.assertTrue(joint_support(rows,'C1','omega512')['passes_pilot_gate'])
        rows[-1]['valid']=False
        self.assertFalse(joint_support(rows,'C1','omega512')['passes_pilot_gate'])
        rows[-1]['valid']=True
        for row in rows: row['scene']=4
        self.assertFalse(joint_support(rows,'C1','omega512')['passes_pilot_gate'])
        rows[-1]['scene']=8; rows[-1]['arm']='C2'
        self.assertFalse(joint_support(rows,'C1','omega512')['passes_pilot_gate'])

    def test_default_matches_legacy_and_mutual_reverses(self):
        import cv2
        cv2.setNumThreads(4)
        module=geometry_module('geometry_confidence')
        import subprocess
        legacy_code=subprocess.check_output(['git','show','57483f93:gtr/geometry/geometry_confidence.py'],text=True)
        scope=dict(__name__=module.__package__+'.legacy',__package__=module.__package__)
        exec(compile(legacy_code,'legacy','exec'),scope)
        rng=np.random.default_rng(20261016)
        a=rng.integers(0,256,(256,320),dtype=np.uint8); b=np.roll(a,7,axis=1)
        for method in ['orb','sift']:
            cv2.setRNGSeed(20261016); old=scope['estimate_background'](a,b,method=method)
            cv2.setRNGSeed(20261016); new=module.estimate_background(a,b,method=method)
            np.testing.assert_array_equal(old['points1'],new['points1'])
            np.testing.assert_array_equal(old['points2'],new['points2'])
            forward=module.estimate_background(a,b,method=method,mutual=True)
            reverse=module.estimate_background(b,a,method=method,mutual=True)
            pairs=lambda x,y:set(map(tuple,np.c_[x,y]))
            self.assertEqual(pairs(forward['points1'],forward['points2']),pairs(reverse['points2'],reverse['points1']))

if __name__=='__main__':unittest.main()

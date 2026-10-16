import unittest
import numpy as np
from omega_modules import geometry_module

class Tests(unittest.TestCase):
    def test_single_real_patch_not_duplicated_or_a_cycle(self):
        P=geometry_module('omega_adapter').PixelMapping
        r=geometry_module('object_patch_regions').object_regions(P(64,64,0,0,64,64,64,64,0,0,64,64),[0,0,16,16])
        self.assertEqual(r['regions']['M0']['selected_count'],1)
        self.assertFalse(r['regions']['M0']['point_correspondence_available'])
        m,_=geometry_module('crossview_point_matcher').match_tokens(np.ones((1,4)),np.ones((1,4)))
        self.assertEqual(m['status'],'POINT_CORRESPONDENCE_UNAVAILABLE');self.assertIsNone(m['hard_cycle_A'])

    def test_padding_ring_and_bbox_edges(self):
        P=geometry_module('omega_adapter').PixelMapping
        p=P(64,32,0,0,64,32,64,32,0,16,64,64)
        module=geometry_module('object_patch_regions');grid=module.patch_lattice(p)
        self.assertEqual(int(grid['valid'].sum()),8)
        r=module.object_regions(p,[8,0,40,32])
        self.assertEqual(r['valid_object_patch_count'],4)
        self.assertTrue(all(not x for x in r['regions']['M2']['is_object_token']))
        for region in r['regions'].values():
            self.assertTrue(np.isfinite(region['original_centers']).all())
            self.assertEqual(len(set(region['patch_ids'])),region['selected_count'])

    def test_control_native_patch14_518_padding(self):
        P=geometry_module('omega_adapter').PixelMapping
        p=P(1920,1080,0,0,1920,1080,518,294,0,112,518,518)
        module=geometry_module('object_patch_regions')
        grid=module.patch_lattice(p,patch_size=14)
        self.assertEqual(grid['grid_shape'],(37,37))
        self.assertEqual(int(grid['valid'].sum()),37*21)
        r=module.object_regions(p,[0,0,1920,1080],patch_size=14)
        self.assertEqual(r['valid_object_patch_count'],37*21)
        self.assertEqual(r['regions']['M0']['selected_count'],16)

    def test_mutual_cycle_redundancy_and_full_denominators(self):
        x=np.eye(4,dtype=np.float32);y=x.copy()
        m,_=geometry_module('crossview_point_matcher').match_tokens(x,y)
        self.assertEqual(m['mutual_count'],4);self.assertEqual(m['hard_cycle_A'],m['mutual_coverage_A'])
        self.assertGreater(m['soft_cycle_A'],.99)
        y=np.tile(x[:1],(4,1));m,_=geometry_module('crossview_point_matcher').match_tokens(x,y)
        self.assertEqual(m['mutual_count'],1);self.assertEqual(m['mutual_coverage_A'],.25)

if __name__=='__main__':unittest.main()

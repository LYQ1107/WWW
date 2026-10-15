import unittest
from summarize_geometry_failures import classify


class Tests(unittest.TestCase):
    def test_low_support_does_not_imply_no_texture_or_overlap(self):
        row=dict(valid='False',inliers='9',inlier_ratio='.6',median_residual='0.001',coverage='.07')
        violations,unknown=classify('traditional_F',row)
        self.assertEqual(violations,['fit_inliers_below20']);self.assertEqual(unknown,[])

    def test_early_return_zero_ratio_is_not_a_measured_bad_residual(self):
        row=dict(valid='False',candidate_inliers='0',validation_matches='25',median_residual='',inlier_ratio='0',coverage='0')
        violations,unknown=classify('neural_F_H',row)
        self.assertIn('required_residual_statistic_unavailable',violations)
        self.assertNotIn('measured_ratio_below0.25_at12px',violations)
        self.assertIn('residual_ratio_unavailable_after_early_return',unknown)


if __name__=='__main__':unittest.main()

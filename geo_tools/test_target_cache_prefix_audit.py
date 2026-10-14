import copy
import unittest
from audit_target_cache_prefix import compare_prefix


class PrefixAuditTest(unittest.TestCase):
    def fixture(self):
        manifest = dict(synchronized_groups=2, causal_online=True, full_video=False,
                        geo_options=dict(REUSE_TARGET_FEATURES=True), geo_failures=0,
                        video_id=1, checkpoint_sha256='a', config_sha256='b',
                        annotation_sha256='c', head_checkpoint_sha256='d', seed=14,
                        source_hashes={'runtime.py':'e'}, validation_corrected_gmt_camera_index=False,
                        validation_reversed_vggt_order=False, head_projection_dimensions=64,
                        planned_update_cost=2, total_vggt_calls=2,
                        target_feature_cache=dict(max_cameras=2, max_targets_per_camera=128,
                                                  peak_payload_bytes=100, stats=dict(reused_targets=1)))
        short = dict(manifest=manifest, predictions=[dict(image_id=1, track_id=8)],
                     schedule=[dict(chosen=True), dict(chosen=False)])
        long = copy.deepcopy(short)
        long['manifest']['synchronized_groups']=3
        long['predictions'].append(dict(image_id=3, track_id=9))
        long['schedule'].append(dict(chosen=False))
        return short, long

    def test_filters_future_but_detects_changed_prefix_id(self):
        short, long = self.fixture()
        self.assertEqual(compare_prefix(short, long, {1,2})['status'], 'PASS')
        long['predictions'][0]['track_id']=9
        self.assertEqual(compare_prefix(short, long, {1,2})['status'], 'FAIL')

    def test_refuses_inactive_reuse_and_changed_model(self):
        short, long = self.fixture()
        short['manifest']['target_feature_cache']['stats']['reused_targets']=0
        with self.assertRaisesRegex(ValueError, 'never reused'):
            compare_prefix(short, long, {1,2})
        short, long = self.fixture()
        long['manifest']['head_checkpoint_sha256']='different'
        with self.assertRaisesRegex(ValueError, 'paired prefix mismatch'):
            compare_prefix(short, long, {1,2})


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch
from audit_tracking_retrieval_prefix import verify_tracking_options,verify_dense_counters


class Tests(unittest.TestCase):
    def test_verified_legacy_no_cache_is_false_but_unknown_or_changed_options_fail(self):
        reference=dict(commit='old',source_hashes={'gtr/config.py':'digest'},geo_options={'FEATURE_WEIGHT':2})
        info=dict(tracking_options={'FEATURE_WEIGHT':2,'REUSE_TARGET_FEATURES':False})
        with patch('audit_tracking_retrieval_prefix.geo_defaults',return_value=({},'digest')):
            self.assertEqual(verify_tracking_options(info,reference),'digest')
            for changed in [{'REUSE_TARGET_FEATURES':True},{'FEATURE_WEIGHT':1},{'UNKNOWN_OPTION':False}]:
                with self.assertRaises(ValueError):verify_tracking_options(dict(tracking_options={**info['tracking_options'],**changed}),reference)
            with self.assertRaises(ValueError):verify_tracking_options(info,dict(reference,source_hashes={'gtr/config.py':'wrong'}))

    def test_actual_dense_encoder_counts_match_declared_backend(self):
        for method,vggt,dino in [('vggt_dense',80,0),('gmt_dense_head',0,0),('dinov2_dense_head',0,40)]:
            stats=dict(updates=40,failures=0,vggt_calls=vggt,dinov2_calls=dino)
            verify_dense_counters(method,stats,80,40)
            for changed in [dict(updates=39),dict(failures=1),dict(vggt_calls=vggt+1),dict(dinov2_calls=dino+1)]:
                with self.assertRaises(ValueError):verify_dense_counters(method,{**stats,**changed},80,40)
        with self.assertRaises(ValueError):verify_dense_counters('dinov2_dense_head',dict(updates=40,failures=0,vggt_calls=0),80,40)

    def test_tracking_modes_cannot_be_combined(self):
        import contextlib,io
        import run_live_retrieval
        for first in ['--tracking-native-prior','--tracking-vggt-dense']:
            with patch('sys.argv',['run_live_retrieval.py','--tag','unit_conflicting_modes',first,'--tracking-head-backbone','dinov2']),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:run_live_retrieval.main()
                self.assertEqual(error.exception.code,2)


if __name__=='__main__':unittest.main()

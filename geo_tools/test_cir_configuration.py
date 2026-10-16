import unittest
import common
from cir_configuration import configuration_hashes
from detectron2.config import get_cfg
from centernet.config import add_centernet_config
from gtr.config import add_gtr_config


class ConfigurationContracts(unittest.TestCase):
    def cfg(self):
        cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg);return cfg
    def test_output_directory_is_operational_only(self):
        a=self.cfg();b=a.clone();b.OUTPUT_DIR='/some/other/run'
        self.assertEqual(configuration_hashes(a,'fixedSHA'),configuration_hashes(b,'fixedSHA'))
    def test_c1_changes_policy_but_not_frozen_gmt(self):
        a=self.cfg();b=a.clone();b.MODEL.CIR.ENABLED=True;b.MODEL.CIR.MODE='c1'
        x,y=configuration_hashes(a,'fixedSHA'),configuration_hashes(b,'fixedSHA')
        self.assertEqual(x['frozen_GMT_config_sha256'],y['frozen_GMT_config_sha256'])
        self.assertNotEqual(x['protocol_config_sha256'],y['protocol_config_sha256'])
    def test_threshold_and_backbone_weight_changes_not_hidden(self):
        a=self.cfg();b=a.clone();b.VIDEO_TEST.OVERLAP_THRESH+=.01
        self.assertNotEqual(configuration_hashes(a,'fixedSHA')['frozen_GMT_config_sha256'],
                            configuration_hashes(b,'fixedSHA')['frozen_GMT_config_sha256'])
        self.assertNotEqual(configuration_hashes(a,'fixedSHA')['frozen_GMT_config_sha256'],
                            configuration_hashes(a,'differentSHA')['frozen_GMT_config_sha256'])

    def test_saved_comparison_preserves_settings_despite_yaml_aliases(self):
        import tempfile
        from pathlib import Path
        from cir_configuration import saved_configuration_hashes
        a=self.cfg();a.merge_from_file(str(common.ROOT/'configs/VISION_test.yaml'))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'first.yaml';q=Path(td)/'reload.yaml'
            p.write_text(a.dump());b=self.cfg();b.merge_from_file(str(p));q.write_text(b.dump())
            self.assertEqual(saved_configuration_hashes(p,'fixedSHA'),saved_configuration_hashes(q,'fixedSHA'))
            b.VIDEO_TEST.OVERLAP_THRESH+=.01;q.write_text(b.dump())
            self.assertNotEqual(saved_configuration_hashes(p,'fixedSHA')['frozen_GMT_config_sha256'],
                                saved_configuration_hashes(q,'fixedSHA')['frozen_GMT_config_sha256'])


if __name__=='__main__':unittest.main()

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


if __name__=='__main__':unittest.main()

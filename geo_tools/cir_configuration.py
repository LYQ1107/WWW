"""Compare scientific GMT settings independently of operational output paths."""
import hashlib
from common import ROOT
from detectron2.config import CfgNode as CN,get_cfg
from centernet.config import add_centernet_config
from gtr.config import add_gtr_config


def configuration_hashes(cfg,checkpoint_sha256):
    normalized=cfg.clone();normalized.defrost()
    normalized.OUTPUT_DIR=''
    normalized.MODEL.WEIGHTS=checkpoint_sha256
    policy=hashlib.sha256(normalized.dump().encode()).hexdigest()
    normalized.MODEL.CIR=CN()
    frozen=hashlib.sha256(normalized.dump().encode()).hexdigest()
    return dict(protocol_config_sha256=policy,frozen_GMT_config_sha256=frozen)


def reference_configuration_hash(checkpoint_sha256):
    cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg)
    cfg.merge_from_file(str(ROOT/'configs/VISION_test.yaml'))
    return configuration_hashes(cfg,checkpoint_sha256)['frozen_GMT_config_sha256']


def saved_configuration_hashes(path,checkpoint_sha256):
    cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg);cfg.merge_from_file(str(path))
    return configuration_hashes(cfg,checkpoint_sha256)

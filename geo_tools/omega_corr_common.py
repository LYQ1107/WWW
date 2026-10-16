"""Independent Omega-Corr runtime; preserve all historical Omega geometry assets."""
from pathlib import Path
from omega_common import ROOT,DATA,NATIVE,jsonl,sha256,save_new,provenance,verified_native
RUNTIME=Path('/data1/liuyeqiang/WWW_omega_corr_runtime_20261016')
OMEGA_RUNTIME=Path('/data1/liuyeqiang/WWW_omega_runtime_20261016')
FIT=(4,5,6,7,8,20,24)
MECHANISM=(9,13,17,21)
DIAGNOSTIC=(12,)


def role(scene):
    if scene in FIT:return 'FIT'
    if scene in MECHANISM:return 'MECHANISM_CHECK'
    if scene in DIAGNOSTIC:return 'EXPOSED_DIAGNOSTIC_ONLY'
    raise ValueError('scene outside registered TRAIN scope')

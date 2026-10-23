#!/usr/bin/env python3
"""Freeze the detector after TRAIN-only quality/integrity admission."""
import argparse
import json
from pathlib import Path
from worldmot.models.person_detector import sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--test-metadata',type=Path,required=True);a=p.parse_args()
    lock_path=a.root/'raw_train_v1/DETECTOR_LOCK.json'
    integrity_path=a.root/'raw_train_integrity_v1.json'
    quality_path=a.root/'train_validation_detection_quality_v1.json'
    lock=json.loads(lock_path.read_text());integrity=json.loads(integrity_path.read_text())
    quality=json.loads(quality_path.read_text());test=json.loads(a.test_metadata.read_text())
    if (integrity['status']!='VERIFIED_COMPLETE_IMMUTABLE_RAW_GMT_SPLIT'
        or len(integrity['scenes'])!=24 or integrity['timestamps']!=28754
        or integrity['cache_lock_sha256']!=sha256_file(lock_path)):
        raise RuntimeError('Complete immutable TRAIN detection gate required')
    if (quality['status']!='COMPLETE_SIX_TRAIN_VALIDATION_DETECTOR_DIAGNOSTIC'
        or quality['TEST_used'] or quality['operating_points']['GMT']!=lock['threshold']):
        raise RuntimeError('TRAIN-only fixed detector quality audit required')
    if test.get('annotations') or len(test['videos'])!=22 or len(test['images'])!=58038:
        raise RuntimeError('Expected full independent GT-free TEST metadata')
    frozen={'status':'GMT_FRONTEND_FROZEN_BEFORE_NEW_TEST_CANDIDATES',
        'TEST_used_for_selection':False,'detector_threshold_rule':'fixed native 0.525; no TEST tuning',
        'TRAIN_detector_lock_sha256':sha256_file(lock_path),'TRAIN_integrity_sha256':sha256_file(integrity_path),
        'TRAIN_quality_sha256':sha256_file(quality_path),'frozen_detector_signature':lock,
        'TEST_metadata_sha256':sha256_file(a.test_metadata),'new_association_TEST_started':False,
        'scope':'Only the detector is admitted for GT-free TEST candidate export; association weights/thresholds require their own later TRAIN freeze.'}
    target=a.root/'detector_frontend_freeze.json'
    if target.exists():
        if json.loads(target.read_text())!=frozen:raise RuntimeError('Existing immutable frontend freeze differs')
    else:
        with target.open('x') as f:f.write(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps({'status':frozen['status'],'manifest':str(target),'sha256':sha256_file(target)}),flush=True)


if __name__=='__main__':main()

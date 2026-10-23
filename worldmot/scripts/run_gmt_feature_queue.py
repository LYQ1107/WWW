#!/usr/bin/env python3
"""Durable full-TRAIN GMT+Omega object-cache worker; one model load per card."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import cache_fulltrain as cache
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.omega_geometry_backend import OmegaGeometryBackend


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scenes',nargs='+',type=int,required=True)
    p.add_argument('--label',required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    state=a.output/(a.label+'_status.json')
    def status(stage,**fields):
        cache.atomic_json(state,{'stage':stage,'pid':os.getpid(),'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'worker_label':a.label,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'assigned_scenes':a.scenes,'training_or_TEST_inference':False,**fields})
    status('INITIALIZING_FROZEN_GMT_CACHE_AND_OMEGA')
    import torch
    torch.set_num_threads(2)
    import cv2;cv2.setNumThreads(1)
    cache.SCHEMA='worldmot_object_v4_gmt_rgb128_detector128_omega256_geometry18_scores2'
    cache.INPUT_CONTRACT='caller_RGB_immutable_GMT_pre_GTA_ROI_F32'
    detector=ImmutableGMTDetectionCache(a.raw_cache)
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='train')
    omega=OmegaGeometryBackend(cache.DEFAULT_OMEGA,cache.DEFAULT_OMEGA_WEIGHTS,device='cuda:0',expose_intermediate_features=True)
    if omega.status!='READY':raise RuntimeError('Frozen Omega backend unavailable: '+str(omega.reason))
    complete=[]
    try:
        for sid in a.scenes:
            prereq=a.raw_cache/f'scene_{sid:02d}'/'manifest.json'
            while not prereq.exists() or json.loads(prereq.read_text()).get('status')!='COMPLETE':
                status('WAITING_FOR_EXACT_RAW_GMT_SCENE',scene=sid,completed_scenes=complete)
                time.sleep(30)
            output_manifest=a.output/f'scene_{sid:02d}'/'manifest.json'
            if output_manifest.exists():
                existing=json.loads(output_manifest.read_text())
                if existing.get('status')=='COMPLETE':
                    if (existing['schema']!=cache.SCHEMA or existing['timestamps']!=len(ds.frames(sid))
                        or existing['detector']['cache_lock_sha256']!=detector.manifest()['cache_lock_sha256']):
                        raise RuntimeError('Existing cache does not match the registered full GMT scene')
                    for shard in existing['shards']:
                        if cache.sha(output_manifest.parent/shard['file'])!=shard['sha256']:
                            raise RuntimeError('Existing GMT object shard changed')
                    complete.append(sid);continue
            status('CACHING_REAL_FULL_TRAIN_SCENE',scene=sid,completed_scenes=complete)
            result=cache.cache_scene(ds,sid,a.output,detector,omega)
            if result['status']!='COMPLETE' or result['timestamps']!=len(ds.frames(sid)):
                raise RuntimeError('Incomplete scene cannot satisfy full TRAIN coverage')
            complete.append(sid)
        status('COMPLETE_ASSIGNED_FULL_TRAIN_SCENES',completed_scenes=complete)
    except BaseException as error:
        status('FAILED_REQUIRES_ENGINEERING_FIX',completed_scenes=complete,error=repr(error))
        raise


if __name__=='__main__':main()

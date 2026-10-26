#!/usr/bin/env python3
"""Byte-preserving local mirror of registered frozen inputs; no new features."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

SCHEMA='worldmot_v4_verified_input_mirror_v1'
R2='worldobject_v3_birthfix_r2_20261025'
RAW='gmt_detector_comparison_20261023'
TRAIN_SCENES=(4,8,12,16,20,24)
TEST_SCENES=tuple(range(1,23))
METHODS=('FULL_WORLD_OBJECT_V3','FULL_NO_GEOMETRY_V3','HEAD_ONLY_NO_GEOMETRY_V3')
WEIGHTS={
    'FULL_WORLD_OBJECT_V3':('cfda40050e89b31dd4480ca4d5010e23067a94ba18ea7d0e2425d5935d3f7fb8','363a0a687466bff7a3c8c93b2589c981ceb69a67e36fa8b58953a2d3b41aaf0d'),
    'FULL_NO_GEOMETRY_V3':('b08badfc634c9652a34366951c93207fa321deb7d1a2b240a76798f11607053f','1796b14b36f9a45381a2ed6067149b746482adceec4b233f9656779ae4c00b1f')}
RETAINED='d4644ff95cbb7f255f70f155055fba6870863a69e7043f91b920e85ce20693dd'


class ChecksumMismatch(RuntimeError):pass


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def safe_relative(value):
    p=Path(value)
    if p.is_absolute() or not p.parts or any(x in ('..','.') for x in p.parts):
        raise ValueError('UNSAFE_MIRROR_RELATIVE_PATH')
    return p


def retry_io(operation,attempts=3):
    for attempt in range(attempts):
        try:return operation()
        except OSError:
            if attempt+1==attempts:raise
            time.sleep(attempt+1)


def copy_verified(source,destination,expected,*,attempts=3):
    """Never expose a partial file or accept bytes with a different checksum."""
    source,destination=Path(source),Path(destination)
    if destination.is_symlink():raise ValueError('MIRROR_FILE_MUST_NOT_BE_SYMLINK')
    if destination.is_file():
        if digest(destination)!=expected:raise ChecksumMismatch('EXISTING_MIRROR_FILE_CHANGED: '+str(destination))
        return False
    destination.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(attempts):
        temporary=destination.with_name(destination.name+f'.partial.{os.getpid()}.{attempt}')
        try:
            h=hashlib.sha256()
            with source.open('rb') as src,temporary.open('xb') as dst:
                for data in iter(lambda:src.read(8<<20),b''):h.update(data);dst.write(data)
                dst.flush();os.fsync(dst.fileno())
            if h.hexdigest()!=expected:raise ChecksumMismatch('ORIGINAL_FROZEN_BYTES_MISMATCH: '+str(source))
            if digest(temporary)!=expected:raise ChecksumMismatch('LOCAL_COPY_CHECKSUM_MISMATCH')
            temporary.chmod(0o400);os.replace(temporary,destination)
            return True
        except OSError:
            if temporary.exists():temporary.unlink()
            if attempt+1==attempts:raise
            time.sleep(attempt+1)
        except ChecksumMismatch:
            # Retain the rejected bytes for diagnosis, without accepting them.
            raise


class PlanBuilder:
    def __init__(self,source,destination,repo):
        self.source,self.destination,self.repo=map(Path,(source,destination,repo));self.entries={}

    def add(self,relative,expected,policy='original_manifest_sha256'):
        if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('EXPECTED_ORIGINAL_SHA256_REQUIRED')
        relative=safe_relative(relative);src=self.source/relative
        size=retry_io(lambda:src.stat().st_size)
        row={'relative':str(relative),'sha256':expected,'bytes':size,'verification_policy':policy}
        if str(relative) in self.entries and self.entries[str(relative)]!=row:raise ValueError('MIRROR_PLAN_CONFLICT')
        self.entries[str(relative)]=row

    def metadata(self,relative,expected=None):
        relative=safe_relative(relative);raw=retry_io(lambda:(self.source/relative).read_bytes())
        actual=hashlib.sha256(raw).hexdigest()
        if expected is not None and actual!=expected:raise ChecksumMismatch('ORIGINAL_METADATA_PROOF_MISMATCH: '+str(relative))
        value=json.loads(raw)
        self.add(relative,actual,'published_proof_sha256' if expected else 'captured_original_metadata_sha256')
        # Small original metadata is staged once; bulk immutable arrays follow.
        copy_verified(self.source/relative,self.destination/relative,actual)
        return value

    def cache(self,base,scenes,*,support):
        total=0
        for scene in scenes:
            folder=Path(base)/f'scene_{scene:02d}';manifest=self.metadata(folder/'manifest.json')
            if manifest.get('status')!='COMPLETE':raise ValueError('COMPLETE_ORIGINAL_CACHE_REQUIRED')
            if support and manifest.get('schema')!='worldobject_omega512_sparse_support_v2_birthfix_v1':raise ValueError('REGISTERED_SUPPORT_SCHEMA_REQUIRED')
            if not support and (manifest.get('GT_in_cache') is not False or manifest.get('track_ids_in_cache') is not False):raise ValueError('GT_FREE_RAW_CACHE_REQUIRED')
            total+=int(manifest['timestamps' if support else 'frames'])
            for shard in manifest['shards']:
                name=safe_relative(shard['file'])
                if len(name.parts)!=1:raise ValueError('SHARD_MUST_BE_SCENE_LOCAL')
                self.add(folder/name,shard['sha256'])
                sidecar=(folder/name).with_suffix('.json')
                if shard.get('sidecar_sha256'):self.add(sidecar,shard['sidecar_sha256'])
                else:self.metadata(sidecar)
            print(json.dumps({'mirror_plan_scene':scene,'cache':str(base),'registered_timestamps':total}),flush=True)
        return total

    def build(self):
        result_dir=self.repo/'worldmot/results'
        gmt_evidence=json.loads((result_dir/'gmt_vggt_official_evidence_20261024.json').read_text())
        coverage=json.loads((result_dir/'worldobject_v3_fulltest_coverage.json').read_text())
        native=next(x for x in gmt_evidence['routes'] if x['method']=='gmt_native')
        gmt_base=Path('fulltrain_20261022/test/gmt_native')
        if not native['prediction_root'].endswith(str(gmt_base)):raise ValueError('REGISTERED_GMT_ROOT_REQUIRED')
        for method,(weight,gate_sha) in WEIGHTS.items():
            folder=Path(R2)/'train'/method;gate=self.metadata(folder/'train_gate.json',gate_sha)
            if gate.get('checkpoint_sha256')!=weight:raise ValueError('REGISTERED_WEIGHT_REQUIRED')
            name=safe_relative(gate['checkpoint'])
            if len(name.parts)!=1:raise ValueError('REGISTERED_WEIGHT_MUST_BE_GATE_LOCAL')
            self.add(folder/name,weight)
        self.metadata(Path(R2)/'audit_snapshots/epoch_archive_index.json')
        self.add(Path(R2)/'audit_snapshots/FULL_WORLD_OBJECT_V3'/f'{RETAINED}.pt',RETAINED)
        self.metadata('fulltrain_20261022/audit/data_audit.json')
        metadata=self.metadata(Path(R2)/'metadata_only_dataset/annotations/train.json')
        if metadata.get('annotations'):raise ValueError('PHYSICAL_METADATA_ONLY_LIVE_INPUT_REQUIRED')
        original_link=self.source/R2/'metadata_only_dataset/train'
        if not original_link.is_symlink():raise ValueError('REGISTERED_LOCAL_RGB_LINK_REQUIRED')
        rgb_target=Path(os.readlink(original_link))
        if rgb_target!=Path('/data/DATASETS/TRACKING/JDE/VisionTrack/train') or not rgb_target.is_dir():raise ValueError('ORIGINAL_LOCAL_RGB_TARGET_REQUIRED')
        link=self.destination/R2/'metadata_only_dataset/train';link.parent.mkdir(parents=True,exist_ok=True)
        if link.is_symlink():
            if Path(os.readlink(link))!=rgb_target:raise ValueError('LOCAL_RGB_LINK_CHANGED')
        elif link.exists():raise ValueError('REFUSE_OVERWRITE_RGB_PATH')
        else:link.symlink_to(rgb_target,target_is_directory=True)
        totals={}
        for split,scenes,expected in (('train',TRAIN_SCENES,5860),('test',TEST_SCENES,29019)):
            raw=Path(RAW)/f'raw_{split}_v1';self.metadata(raw/'DETECTOR_LOCK.json')
            a=self.cache(raw,scenes,support=False);b=self.cache(Path(R2)/f'cache_{split}_v2',scenes,support=True)
            if a!=expected or b!=expected:raise ValueError('EXACT_REGISTERED_FULL_VIDEO_SCOPE_REQUIRED')
            totals[split]={'scenes':list(scenes),'timestamps':expected}
        proofs={'GMT_NATIVE':{x['scene']:x['sha256'] for x in native['manifests']}}
        proofs.update({m:{x['scene']:x['manifest_sha256'] for x in coverage['methods'][m]['scenes']} for m in METHODS})
        for method in ('GMT_NATIVE',*METHODS):
            total=0;base=gmt_base if method=='GMT_NATIVE' else Path(R2)/'predictions'/method
            for scene in TEST_SCENES:
                folder=base/f'scene_{scene:02d}';manifest=self.metadata(folder/'manifest.json',proofs[method][scene])
                if manifest.get('status')!='COMPLETE' or manifest.get('GT_in_inference') is not False:raise ValueError('FROZEN_GT_FREE_PREDICTIONS_REQUIRED')
                total+=int(manifest.get('timestamps',manifest.get('frames')))
                self.add(folder/'tracks.jsonl',manifest.get('tracks_sha256',manifest.get('prediction_sha256')))
            if total!=29019:raise ValueError('EXACT_ALL22_PREDICTIONS_REQUIRED')
        return {'schema':SCHEMA,'source_root':str(self.source),'destination_root':str(self.destination),
            'registered_scope':totals,'new_features_or_geometry_produced':False,'entries':list(self.entries.values())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--reserve-GiB',type=float,default=25);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    script_sha=digest(__file__)
    if a.source.absolute()==a.destination.absolute() or a.destination.absolute().is_relative_to(a.source.absolute()):raise ValueError('LOCAL_MIRROR_MUST_BE_SEPARATE_FROM_ORIGINAL')
    a.destination.mkdir(parents=True,exist_ok=True,mode=0o700);repo=Path(__file__).resolve().parents[2]
    marker=a.destination/'MIRROR_OWNER.json';identity={'schema':SCHEMA,'source_root':str(a.source),'destination_root':str(a.destination)}
    if marker.exists():
        if json.loads(marker.read_text())!=identity:raise ValueError('DIFFERENT_MIRROR_OWNER')
    elif any(a.destination.iterdir()):raise ValueError('FRESH_DESTINATION_OR_MATCHING_MARKER_REQUIRED')
    else:atomic_json(marker,identity)
    a.destination.chmod(0o700)
    plan_path=a.destination/'mirror_plan.json';receipt_path=a.destination/'mirror_receipt.json'
    try:
        if plan_path.exists():
            plan=json.loads(plan_path.read_text())
            if any(plan[k]!=v for k,v in identity.items()):raise ValueError('MIRROR_PLAN_IDENTITY_CHANGED')
        else:
            plan=PlanBuilder(a.source,a.destination,repo).build();atomic_json(plan_path,plan)
        entries=plan['entries'];needed=sum(x['bytes'] for x in entries if not (a.destination/safe_relative(x['relative'])).is_file())
        free=shutil.disk_usage(a.destination).free;reserve=int(a.reserve_GiB*(1<<30))
        if needed+reserve>free:raise RuntimeError('INSUFFICIENT_LOCAL_SPACE_FOR_REGISTERED_MIRROR_AND_RESERVE')
        receipt={'schema':SCHEMA,'status':'PLANNED' if a.plan_only else 'COPYING','plan_sha256':digest(plan_path),
            'staging_script_sha256':script_sha,
            'source_root':str(a.source),'destination_root':str(a.destination),'files':len(entries),'total_bytes':sum(x['bytes'] for x in entries),
            'additional_bytes_needed':needed,'initial_local_free_bytes':free,'reserved_free_bytes':reserve,
            'registered_scope':plan['registered_scope'],'new_features_or_geometry_produced':False,'GT_used_for_copy_selection':False}
        atomic_json(receipt_path,receipt);print(json.dumps(receipt),flush=True)
        if a.plan_only:return
        copied=skipped=0;started=time.perf_counter()
        for index,item in enumerate(entries):
            if shutil.disk_usage(a.destination).free<reserve:raise RuntimeError('LOCAL_SPACE_RESERVE_REACHED')
            relative=safe_relative(item['relative'])
            if copy_verified(a.source/relative,a.destination/relative,item['sha256']):copied+=1
            else:skipped+=1
            if (index+1)%20==0 or index+1==len(entries):
                receipt.update(verified_files=index+1,copied_files=copied,already_verified_files=skipped,
                    elapsed_wall_s=time.perf_counter()-started,last_verified_relative=str(relative))
                atomic_json(receipt_path,receipt);print(json.dumps({'mirror_progress':index+1,'files':len(entries),'relative':str(relative)}),flush=True)
        if digest(__file__)!=script_sha:raise RuntimeError('STAGING_SOURCE_CHANGED_DURING_COPY')
        receipt.update(status='VERIFIED',all_original_file_checksums_verified=True,verified_UTC=datetime.now(timezone.utc).isoformat())
        atomic_json(receipt_path,receipt);print(json.dumps(receipt),flush=True)
    except Exception as error:
        atomic_json(receipt_path,{'schema':SCHEMA,'status':'FAILED','error_type':type(error).__name__,'error':str(error),
            'verified_UTC':datetime.now(timezone.utc).isoformat(),'original_inputs_modified':False})
        raise


if __name__=='__main__':main()

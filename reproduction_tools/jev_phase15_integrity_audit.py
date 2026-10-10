"""Rehash every protected historical checkpoint before final delivery."""
import time
from jev_phase15_common import *


def main():
    protect();manifest=read(REPORTS/'SOURCE_AND_CHECKPOINT_MANIFEST.json');begin=time.monotonic();verified=[]
    for index,item in enumerate(manifest['checkpoints']):
        assert sha(item['path'])==item['SHA256'],item['path'];verified.append(item)
        if index%25==0:
            save(OUT/'final_integrity/PROGRESS.json',dict(status='HASHING_PROTECTED_PRIOR_WEIGHTS',done=index+1,total=len(manifest['checkpoints']),seconds=time.monotonic()-begin))
            print('PHASE15_PRIOR_HASH_VERIFIED',index+1,flush=True)
    reports=read(REPORTS/'PHASE14_FROZEN_EVIDENCE.json')['protected_report_SHA256']
    for path,digest in reports.items():assert sha(ROOT/path)==digest
    report=dict(status='PASS',binding=binding(seed=20261009,checkpoints=verified,dataset=ref(ANNOTATIONS),
        evaluator='complete SHA256 recheck of frozen prior weights and reports',scope='all V-XIV protected assets unchanged'),
        checkpoint_count=len(verified),protected_report_count=len(reports),source_manifest=ref(REPORTS/'SOURCE_AND_CHECKPOINT_MANIFEST.json'),
        seconds=time.monotonic()-begin,all_protected_bytes_unchanged=True)
    save(OUT/'final_integrity/RESULT.json',report);save(REPORTS/'FINAL_PRIOR_INTEGRITY.json',report)


if __name__=='__main__':main()

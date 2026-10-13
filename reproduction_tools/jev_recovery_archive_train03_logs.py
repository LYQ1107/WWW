"""Lossless inactive TRAIN03 log archival with the existing verified utility."""
import sys,time,pathlib,json,shutil
sys.path.insert(0,'/home/liuyeqiang/WWW_jev_master_audit/reproduction_tools')
from jev_master_storage_cleanup import fingerprint,archive_one,save

ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=pathlib.Path('/data1/liuyeqiang/WWW_storage_maintenance/20261010/home_runtime/WWW_jev_phase6_runtime/20261008/heldouts/video03/G2')
STORE=pathlib.Path('/data1/liuyeqiang/WWW_storage_maintenance/20261013_train03_logs')

if __name__=='__main__':
    STORE.mkdir(exist_ok=True);ledger_path=STORE/'ARCHIVE_LEDGER.json'
    assert not ledger_path.exists(),'Preserve prior archival attempt'
    ledger=dict(status='RUNNING',scope='Inactive legacy TRAIN03 logs only, not metrics or model checkpoints',files=[],free_before_GiB=shutil.disk_usage('/data1').free/2**30)
    save(ledger_path,ledger)
    for path in [BASE/'online_decisions.jsonl',BASE/'tracking_decisions/jev.json']:
        original=fingerprint(path);assert time.time_ns()-original['mtime_ns']>=48*3600*10**9
        archive_one(path,original,ledger,ledger_path)
    ledger.update(status='COMPLETE_LOSSLESS_ARCHIVAL',freed_allocated_GiB=sum(f['freed_allocated_bytes'] for f in ledger['files'])/2**30,free_after_GiB=shutil.disk_usage('/data1').free/2**30)
    save(ledger_path,ledger);save(ROOT/'reports/JEV_RECOVERY/TRAIN03_LOG_ARCHIVE_20261013.json',ledger)
    print('TRAIN03_ARCHIVE_COMPLETE_FREED_GiB',ledger['freed_allocated_GiB'],flush=True)

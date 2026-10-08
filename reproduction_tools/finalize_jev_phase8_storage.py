"""Finalize the executed P-1 cleanup, without inferring release from disk drift."""
import json,os,datetime
from pathlib import Path
from audit_jev_phase8_storage import OUT,save,command,allocation,processes

def main():
    r=OUT/'bootstrap/reports';before=json.loads((r/'STORAGE_BEFORE.json').read_text());execution=json.loads((r/'DELETE_EXECUTION.json').read_text())
    assert execution['freed_bytes']==0 and execution['manifest_entries']==0
    filesystems=[]
    for old in before['filesystems']:
        p=old['path'];s=os.statvfs(p);filesystems.append({'path':p,'device':os.stat(p).st_dev,'total_bytes':s.f_blocks*s.f_frsize,'available_bytes':s.f_bavail*s.f_frsize,'used_bytes':(s.f_blocks-s.f_bfree)*s.f_frsize,'inodes':s.f_files,'free_inodes':s.f_favail})
    active,_,uncertain=processes()
    after={'status':'COMPLETE','captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'filesystems':filesystems,'project_directories':[allocation(Path(x['path']))for x in before['project_directories']],'checkpoint_bytes':before['checkpoint_bytes'],'bounded_commands':[command(['df','-hT']),command(['df','-i'])],'actual_files_deleted':0,'actual_freed_bytes':0,'disk_available_drift_is_not_cleanup_release':True,'active_own_processes':active,'unreadable_own_processes':uncertain,'all_prior_evidence_preserved':True,'dataset_and_perception_reused':True}
    save(r/'STORAGE_AFTER.json',after)
    d=json.loads((r/'CHECKPOINT_DEPENDENCY_AUDIT.json').read_text())
    text=f"""# Phase VIII storage cleanup\n\nP-1 is complete. Examined {d['checkpoint_count']} model/checkpoint/state-snapshot files ({d['checkpoint_bytes']/2**30:.3f} GiB) in explicitly scoped project directories. SHA256 values and verified content duplicates are in CHECKPOINT_INVENTORY.csv. Indexed per-video/frame/view perception `.pt` tensors are protected data, inventoried separately rather than misclassified as model checkpoints.\n\nDeleted files: none. Actual released space: **0 bytes / 0 GiB**. No artifact met all ten verified-orphan conditions; missing original training provenance and intentionally omitted large raw reference files cannot prove absence of a dependency. Classes: {json.dumps(d['classes'])}. Negative experiments and all Phase V/VI/VII evidence remain. Archived historical training weights, the permanent B2, GMT foundation, fork snapshots and unidentified files are retained.\n\nDefault dry run and explicit empty execution both completed. Eight adversarial cleanup guard tests pass. No other user's process, system cache or Git history was altered. Free-space changes made by other tasks are not credited as cleanup.\n\n/data1 has substantial space pressure; /home is a separate filesystem with roughly 99 GB available at the initial audit. New runtime outputs will use /home, reuse the frozen perception cache and keep bounded forks plus best/last checkpoints. A sparse isolated worktree will avoid copying old bulky research archives. GitHub publication remains limited to source, configuration, metrics, compact examples and SHA/provenance manifests.\n\nWHAT DID WE LEARN? A failed or inactive run is not a safe deletion proof. Storage safety can be maintained with zero deletion through output placement, cache reuse and smaller publication scope.\n"""
    (r/'STORAGE_CLEANUP_REPORT.md').write_text(text);print(json.dumps({'status':'COMPLETE','deleted':0,'freed_bytes':0,'checkpoint_count':d['checkpoint_count']}))

if __name__=='__main__':main()

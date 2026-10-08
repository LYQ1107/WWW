"""Exact-path cleanup. Dry-run by default; unproven entries are refused."""
import argparse, json, os, stat
from pathlib import Path
from audit_jev_phase8_storage import BASE, B2, B2_SHA, FOUNDATION, OUT, processes, reference_corpus, scopes, sha, save

PROOFS = ('no_active_task','no_registered_task','no_resume_dependency','no_retained_evidence_dependency',
          'not_negative_experiment_model','verified_redundant_copy','not_initialization_dependency',
          'known_origin_and_reason','second_reference_check_passed','replay_preserved')

def protected(path):
    return path in (B2,FOUNDATION) or path.name=='model_20000.pth' or any(
        token in str(path) for token in ['/WWW_jev_phase5/','/WWW_jev_phase6/','/WWW_jev_phase7/',
                                        '/WWW_jev_phase5_runtime/','/WWW_jev_phase6_runtime/','/WWW_jev_phase7_runtime/'])

def validate(entry,roots,corpus,opened,uncertain):
    path=Path(entry['path']);assert path.is_absolute() and '..'not in path.parts,'path must be exact absolute'
    assert any(root==path.parent or root in path.parents for root in roots),'outside explicitly inventoried roots'
    assert path.exists()and not path.is_symlink(),'missing or symlink'
    assert all(not ancestor.is_symlink()for ancestor in path.parents),'symlink ancestor'
    assert not protected(path),'protected path'
    assert entry['classification']=='C_VERIFIED_ORPHAN','not a verified orphan'
    assert all(entry['proofs'].get(key)is True for key in PROOFS),'missing one of ten deletion proofs'
    assert entry.get('source_commit')not in [None,'UNKNOWN',''],'origin is unknown'
    assert entry.get('reason'),'missing deletion reason'
    assert not uncertain,'process access uncertainty prevents absence-of-use proof'
    assert not opened.get(str(path)),'active open file'
    info=path.stat();assert stat.S_ISREG(info.st_mode),'not a regular file'
    assert info.st_uid==os.getuid(),'not owned by current user'
    assert info.st_size==entry['size_bytes']and sha(path)==entry['sha256'],'file changed'
    assert entry['sha256']!=B2_SHA,'protected B2 content'
    replica=Path(entry['verified_replica']);assert replica!=path and replica.is_file()and sha(replica)==entry['sha256'],'verified remaining copy required'
    for source,text in corpus:
        assert str(path)not in text and path.name not in text,('new retained reference',source)
    return info

def execute(manifest,dry_run=True):
    roots=[Path(p)for p in manifest['authorized_scope_roots']];rows=[];freed=0
    for entry in manifest['entries']:
        try:
            trees,current=scopes();corpus,omissions,_=reference_corpus(trees,current)
            assert not omissions,'incomplete reference corpus cannot establish safe orphan status'
            _,opened,uncertain=processes()
            first=validate(entry,roots,corpus,opened,uncertain)
            if not dry_run:
                # Second fresh dependency/process/hash check immediately before exact-path unlink.
                corpus,omissions,_=reference_corpus(trees,current);assert not omissions
                _,opened,uncertain=processes();second=validate(entry,roots,corpus,opened,uncertain)
                assert(first.st_dev,first.st_ino,first.st_size,first.st_mtime_ns)==(second.st_dev,second.st_ino,second.st_size,second.st_mtime_ns),'path replaced during verification'
                Path(entry['path']).unlink();freed+=entry['size_bytes']
            rows.append({'path':entry['path'],'status':'DRY_RUN_APPROVED'if dry_run else'DELETED','bytes':entry['size_bytes']})
        except Exception as error:rows.append({'path':entry.get('path'),'status':'REFUSED','reason':str(error)})
    return {'status':'COMPLETE_NO_VERIFIED_ORPHANS'if not rows else'COMPLETE_WITH_REFUSALS'if any(r['status']=='REFUSED'for r in rows)else'COMPLETE',
            'dry_run':dry_run,'files':rows,'freed_bytes':freed,'manifest_entries':len(manifest['entries'])}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=OUT/'bootstrap/reports/SAFE_DELETE_MANIFEST.json')
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text());result=execute(manifest,not args.execute)
    save(args.manifest.parent/('DELETE_EXECUTION.json'if args.execute else'DELETE_DRY_RUN.json'),result)
    print(json.dumps(result))

if __name__=='__main__':main()

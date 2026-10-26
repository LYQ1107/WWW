from pathlib import Path
import hashlib
import json
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from stage_worldobject_v4_inputs import copy_verified,digest,safe_relative,ChecksumMismatch
from resume_worldobject_v4_after_nas import verify_input_mirror
from worldmot.models.world_object_v4_checkpoint import retained_archive_path,RETAINED_EPOCH1_SHA


def test_mirror_preserves_bytes_and_skips_a_verified_existing_copy(tmp_path):
    source=tmp_path/'original';source.write_bytes(bytes(range(256))*1000)
    destination=tmp_path/'mirror/input.npz';expected=digest(source)
    assert copy_verified(source,destination,expected)
    assert source.read_bytes()==destination.read_bytes() and digest(destination)==expected
    assert not copy_verified(source,destination,expected)
    assert source.stat().st_mode&0o200 and not destination.stat().st_mode&0o222


def test_wrong_original_or_tampered_mirror_is_never_accepted(tmp_path):
    source=tmp_path/'original';source.write_bytes(b'frozen')
    destination=tmp_path/'mirror/input'
    with pytest.raises(ChecksumMismatch):copy_verified(source,destination,'0'*64)
    assert not destination.exists() and source.read_bytes()==b'frozen'
    destination.parent.mkdir(exist_ok=True);destination.write_bytes(b'tampered')
    with pytest.raises(ChecksumMismatch):copy_verified(source,destination,digest(source))
    assert destination.read_bytes()==b'tampered'


def test_transient_read_error_retries_without_exposing_partial_input(tmp_path,monkeypatch):
    source=tmp_path/'original';source.write_bytes(b'exact immutable array')
    destination=tmp_path/'mirror/input';expected=digest(source);original=Path.open;calls=[]
    def opening(path,*args,**kwargs):
        if path==source:
            calls.append(1)
            if len(calls)==1:raise OSError(5,'synthetic transient I/O')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opening)
    assert copy_verified(source,destination,expected) and len(calls)==2
    assert digest(destination)==expected and not list(destination.parent.glob('*.partial.*'))


@pytest.mark.parametrize('value',['../original','/absolute/input','nested/../../input'])
def test_mirror_refuses_parent_escape_and_absolute_paths(value):
    with pytest.raises(ValueError):safe_relative(value)


def test_retained_archive_relocation_keeps_exact_registered_checkpoint(tmp_path):
    suffix=Path('audit_snapshots/FULL_WORLD_OBJECT_V3')/(RETAINED_EPOCH1_SHA+'.pt')
    record={'path':str(Path('/old_NAS/R2')/suffix),'sha256':RETAINED_EPOCH1_SHA}
    assert retained_archive_path(tmp_path,record)==tmp_path/suffix
    for wrong in ('different.pt','../different.pt'):
        with pytest.raises(ValueError):retained_archive_path(tmp_path,{**record,'path':wrong})
    with pytest.raises(ValueError):retained_archive_path(tmp_path,{**record,'sha256':'0'*64})


def test_controller_rejects_incomplete_or_tampered_mirror_receipt(tmp_path):
    plan=tmp_path/'mirror_plan.json';plan.write_text('{"synthetic_unit_plan": true}\n')
    receipt=tmp_path/'mirror_receipt.json'
    record={'schema':'worldmot_v4_verified_input_mirror_v1','status':'VERIFIED',
        'all_original_file_checksums_verified':True,'files':1,'verified_files':1,
        'registered_scope':{'train':{'scenes':[4,8,12,16,20,24],'timestamps':5860},
            'test':{'scenes':list(range(1,23)),'timestamps':29019}},
        'destination_root':str(tmp_path),'source_root':'/synthetic/original',
        'plan_sha256':digest(plan)}
    receipt.write_text(json.dumps(record))
    assert verify_input_mirror(receipt,tmp_path/'R2')['transport']=='VERIFIED_LOCAL_BYTE_MIRROR'
    for field,value in [('status','COPYING'),('verified_files',0),('all_original_file_checksums_verified',False),
                        ('plan_sha256','0'*64),('destination_root',str(tmp_path/'wrong'))]:
        receipt.write_text(json.dumps({**record,field:value}))
        with pytest.raises(ValueError):verify_input_mirror(receipt,tmp_path/'R2')

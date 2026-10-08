"""Verify completed archives, portable records, tracked artifacts and protected anchors."""
import gzip,hashlib,json,subprocess
from jev_phase7_common import *

def stream_sha(reader):
    digest=hashlib.sha256()
    for chunk in iter(lambda:reader.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    protect();tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
    checks=[];source_files=archive_files=decompressed=0;byte_count=0;missing=[]
    for path in sorted((REPORTS/'evidence').rglob('ARCHIVE_MANIFEST.json')):
        m=json.loads(path.read_text());assert m['status']=='VERIFIED';rel=str(path.relative_to(ROOT))
        if rel not in tracked:missing.append(rel)
        for r in m['files']:
            archive=path.parent/r['archive'];assert archive.is_file(),archive
            assert sha(archive)==r['archive_sha256'],archive
            source=Path(r['source']);assert source.is_file()and sha(source)==r['source_sha256'],source
            if archive.name.endswith('.gz')and not source.name.endswith('.gz'):
                with gzip.open(archive,'rb')as stream:assert stream_sha(stream)==r['source_sha256'],archive
                decompressed+=1
            elif archive.suffix=='.gz':
                # Read to EOF to reject truncated gzip, without pretending compressed SHA equals plaintext SHA.
                with gzip.open(archive,'rb')as stream:stream_sha(stream)
            archive_files+=1;source_files+=1;byte_count+=archive.stat().st_size
            rel=str(archive.relative_to(ROOT))
            if rel not in tracked:missing.append(rel)
        checks.append({'manifest':str(path.relative_to(ROOT)),'manifest_sha256':sha(path),'files':len(m['files'])})
    required_docs=['PHASE7_RESEARCH_GOAL.md','PHASE7_SOURCE_AUDIT.md','PHASE7_RELATED_WORK_MATRIX.md','PHASE7_REVIEWER_NOVELTY_RISKS.md',
        'PHASE7_CAUSAL_DATASET_DESIGN.md','PHASE7_ARCHITECTURE_DESIGN.md','PHASE7_FAIR_BASELINE_PROTOCOL.md','PHASE7_FINAL_RESEARCH_REPORT.md',
        'PHASE7_REASSOCIATION_ATTRIBUTION.md','PHASE7_LEARNABILITY_AUDIT.md','NATIVE_CAUSAL_MINISET_V1_SCHEMA.md']
    required_reports=['LEARNABILITY_AUDIT.json','CAUSAL_DATASET_AUDIT.json','NATIVE_TRANSITION_PARITY.json','ARCHITECTURE_ABLATION.json',
        'HELDOUT_RESULTS.json','FINAL_GO_NO_GO.json','PHASE7_REASSOCIATION_ABLATION.json','PHASE7_BYTETRACK_COMPARISON.json',
        'PHASE7_MATCH_MECHANISM_GO_NO_GO.json','NATIVE_CAUSAL_MINISET_V1_MANIFEST.json','NATIVE_CAUSAL_MINISET_V1_AUDIT.json']
    for p in [ROOT/'docs'/n for n in required_docs]+[REPORTS/n for n in required_reports]:
        assert p.exists()
        if str(p.relative_to(ROOT))not in tracked:missing.append(str(p.relative_to(ROOT)))
    from jev_phase7_miniset import NativeCausalMiniSet
    for role in ('train','validation'):
        p=REPORTS/'miniset_v1'/f'{role}.jsonl.gz';assert len(NativeCausalMiniSet(p,role))==16
        if str(p.relative_to(ROOT))not in tracked:missing.append(str(p.relative_to(ROOT)))
    modified_production=subprocess.check_output(['git','diff',BASE,'--name-only','--','gtr'],cwd=ROOT,text=True).strip()
    assert not modified_production,'production core changed'
    result={'status':'PASS'if not missing else 'FAIL_MISSING_TRACKED_ARTIFACTS','archive_manifests':checks,'manifests':len(checks),
        'source_files_verified':source_files,'archive_files_verified':archive_files,'decompressed_files_verified':decompressed,
        'archive_bytes':byte_count,'missing_tracked':missing,'permanent_B2_sha256':sha(B2),'PhaseV_HEAD':PHASE5,'PhaseVI_HEAD':BASE,
        'production_gtr_modified':False,'full24_started':False,'official_test_read':False,'portable_train_val_records':32,
        'remote_sync_status':'verify ls-remote after final push; local integrity does not imply remote publication','binding':binding()}
    save(REPORTS/'PUBLICATION_INTEGRITY.json',result)
    assert not missing,missing[:20]
    protect();print(json.dumps({k:v for k,v in result.items()if k not in ['archive_manifests','binding']}))

if __name__=='__main__':main()

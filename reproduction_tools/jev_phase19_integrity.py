"""Verify published evidence, actual source actors, and every new perception payload."""
from jev_phase19_common import *


def main():
    protect();checked={};documents=[];actor_sources=set()
    def verify(p,digest):
        p=Path(p);assert p.is_absolute(),p
        actual=checked.setdefault(str(p),sha(p)) if str(p) not in checked else checked[str(p)]
        assert actual==digest,('evidence hash mismatch',str(p),actual,digest)
    def walk(x):
        if isinstance(x,dict):
            if 'path' in x and 'SHA256' in x:verify(x['path'],x['SHA256'])
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    def audit(p):
        x=read(p);walk(x);documents.append(str(p))
        for field in ['binding','recovery_binding']:
            b=x.get(field)
            if not b or not b.get('preregistration'):continue
            actor=Path(b['preregistration']['path']).parents[2]
            for name,digest in b.get('source_SHA256',{}).items():verify(actor/name,digest)
            actor_sources.add(str(actor))
    for p in sorted(REPORTS.glob('*.json')):
        if p.name!='ARTIFACT_SHA_MANIFEST.json':audit(p)
    for name in ['cache_queue_v1','intervention_queue_v1','census_queue_v1','census_queue_v2']:
        audit(OUT/name/'RESULT.json')
    records=0
    for video in SCOPE:
        audit(census_case(video)/'RESULT.json')
        case=OUT/'stage1_cache_v1'/f'video{video:02d}';audit(case/'RESULT.json')
        for line in (case/'index.jsonl').read_text().splitlines():
            r=json.loads(line);assert r['video_id']==video
            verify(case/r['record'],r['record_sha256'].removeprefix('sha256:'));records+=1
    assert records==28924
    branch_count=0
    for path in sorted((OUT/'interventions').glob('video*/RESULT.json')):
        audit(path)
        for b in read(path)['branches']:audit(Path(b['path']));branch_count+=1
    assert branch_count==66
    for p in (OUT/'interventions').glob('video*/PREFIX_*.json'):audit(p)
    oldroot=Path('/home/liuyeqiang/WWW_jev_phase18')
    old=read(oldroot/'reports/JEV_PHASE18/ARTIFACT_SHA_MANIFEST.json')
    for name,digest in old['files'].items():verify(oldroot/name,digest)
    evidence=read(MASTER/'EVIDENCE_REGISTRY.json')
    walk(evidence)
    existing={r['path'] for r in evidence['entries']}
    for p in sorted(REPORTS.glob('*.json')):
        if p.name!='ARTIFACT_SHA_MANIFEST.json' and str(p) not in existing:evidence['entries'].append(ref(p))
    save(MASTER/'EVIDENCE_REGISTRY.json',evidence)
    names=subprocess.check_output(['git','diff',BASE,'--name-only'],cwd=ROOT,text=True).splitlines()
    names+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    files={p:sha(ROOT/p) for p in sorted(set(names)) if p!='reports/JEV_PHASE19/ARTIFACT_SHA_MANIFEST.json' and (ROOT/p).is_file()}
    assert all(not p.endswith(('.pth','.pt','.gz','.xz','.mp4')) for p in files),'binary in publication'
    save(REPORTS/'ARTIFACT_SHA_MANIFEST.json',dict(status='PASS',files=files,
        referenced_evidence_files_verified=len(checked),actual_source_actors=sorted(actor_sources),
        frozen_Phase18_publication_files_verified=len(old['files']),cache_payloads_verified=records,
        complete_native_branches_verified=branch_count,large_artifacts_server_only=True,
        master_scientific_goal_complete=False,documents_checked=len(documents),
        runtime_GiB=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())/2**30))
    print(json.dumps(dict(status='PASS',evidence_files=len(checked),cache_payloads=records,branches=branch_count,publication_files=len(files))))


if __name__=='__main__':main()

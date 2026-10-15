"""Read-only TRAIN objective applicability census; no video/GT rematching."""
import argparse,gzip,json,time
from pathlib import Path
import torch
from pirga_indexed_store import sha
from pirga_training_cache import NativeTrainingCache


def run(protocol_path,out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();torch.set_num_threads(1);p=json.loads(protocol_path.read_text())
    if sha(p['admission']['path'])!=p['admission']['sha256']:raise ValueError('Sealed admission changed')
    ad=json.loads(Path(p['admission']['path']).read_text())
    if sha(ad['supervision']['path'])!=ad['supervision']['sha256'] or sha(p['common_training_domain']['path'])!=p['common_training_domain']['sha256']:raise ValueError('Frozen TRAIN supervision/domain changed')
    common={tuple(k) for k in json.loads(Path(p['common_training_domain']['path']).read_text())};extra=[]
    with gzip.open(ad['supervision']['path'],'rt') as f:
        for line in f:
            row=json.loads(line);key=tuple(row['event_key']);person=row['GT_person']
            positive=[i for i,v in enumerate(row['candidate_persons']) if person is not None and v==person]
            if positive and key not in common:extra.append((key,positive,row['actual_primary_state']))
    cache=NativeTrainingCache(p['cache']['path'],p['cache']['sha256']);events=[]
    for key,positive,state in extra:
        b=cache.batch(key);counts=torch.bincount(b.owners,minlength=len(b.native_scores))
        events.append(dict(key=list(key),original_state=state,pure_positive_candidates=len(positive),positive_strict_past_support=sum(int(counts[i]) for i in positive)))
    result=dict(status='NATIVE_MARGIN_COMMON_DOMAIN_SCOPE_AUDIT',common_queries=len(common),pure_positive_outside_common=len(events),
        outside_common_with_controllable_positive_history=sum(r['positive_strict_past_support']>0 for r in events),events=events,
        qualification='Read-only sealed TRAIN supervision applicability audit, not training or DEV selection. Current fixed fits unchanged. Extra positive margin targets require a separately qualified coverage supplement.',
        GPU_wall_hours=0,wall_seconds=time.monotonic()-start,source_sha256=sha(__file__),protocol_sha256=sha(protocol_path))
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='events'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.protocol,a.out)

"""Retain small Phase VI evidence with source and archived SHA bindings."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

from jev_phase6_common import OUT, REPORTS, sha, save, protect_anchor


def projection(row):
    """Keep complete controller input and action; trim only oversized diagnostic candidate arrays."""
    for context in (row,row.get('context',{})):
        ids=context.get('candidate_track_ids');scores=context.get('candidate_scores')
        if ids is not None and len(ids)>8:
            context['archive_original_candidate_count']=len(ids)
            context['candidate_track_ids']=ids[:8]
            if scores is not None:context['candidate_scores']=scores[:8]
    return row


def archive(relative):
    protect_anchor()
    source = (OUT / relative).resolve()
    if OUT not in source.parents or not source.is_dir():
        raise ValueError(source)
    target = REPORTS / 'evidence' / relative
    entries = []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(source)
        # TRAIN GT copies and formatter intermediates are reproducible from
        # the declared dataset; retain evaluator outputs and actual raw runs.
        if any(part in {'eval_dataset', 'prepared', 'snapshots','committed_transitions'} for part in rel.parts) or path.suffix == '.npy':
            continue
        dest = target / rel
        if path.suffix in {'.json','.jsonl'} and path.stat().st_size>512*1024*1024:
            dest=dest.with_name(dest.name+'.controller-input-projection.jsonl.gz');dest.parent.mkdir(parents=True,exist_ok=True)
            count=0
            with dest.open('wb') as raw:
                with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as writer:
                    if path.suffix=='.jsonl':
                        with path.open() as reader:
                            for line in reader:
                                row=projection(json.loads(line));writer.write((json.dumps(row,sort_keys=True)+'\n').encode());count+=1
                    else:
                        rows=json.loads(path.read_text())
                        if not isinstance(rows,list):raise ValueError('oversized diagnostic projection requires records')
                        for row in rows:
                            writer.write((json.dumps(projection(row),sort_keys=True)+'\n').encode());count+=1
            entries.append({'source':str(path),'archive':str(dest.relative_to(REPORTS)),
                'source_sha256':sha(path),'archive_sha256':sha(dest),'source_bytes':path.stat().st_size,
                'archive_bytes':dest.stat().st_size,'records':count,
                'archive_format':'diagnostic projection; not a lossless copy of full candidate arrays',
                'preserved':'every original 64D controller input, legal action, selected action, proposal metadata; exact predictions and evaluator outputs separately retained',
                'omitted':'diagnostic candidate IDs/scores after first8; original count retained; full source remains at declared runtime path with SHA'})
            continue
        if path.suffix=='.jsonl' and path.stat().st_size>128*1024*1024:
            source_digest=sha(path);line_start=1;part=0
            with path.open('rb') as reader:
                while True:
                    lines=[]
                    for _ in range(4000):
                        line=reader.readline()
                        if not line:break
                        lines.append(line)
                    if not lines:break
                    chunk=dest.with_name(dest.name+f'.part{part:05d}.gz');chunk.parent.mkdir(parents=True,exist_ok=True)
                    with chunk.open('wb') as raw:
                        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as writer:writer.writelines(lines)
                    entries.append({'source':str(path),'archive':str(chunk.relative_to(REPORTS)),
                        'source_sha256':source_digest,'archive_sha256':sha(chunk),
                        'source_bytes':path.stat().st_size,'archive_bytes':chunk.stat().st_size,
                        'jsonl_lines':[line_start,line_start+len(lines)-1],
                        'reconstruction':'concatenate decompressed numbered parts in ascending order'})
                    line_start+=len(lines);part+=1
            continue
        if path.suffix in {'.jsonl', '.json'} and path.stat().st_size > 512 * 1024:
            dest = dest.with_name(dest.name + '.gz')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with path.open('rb') as reader, dest.open('wb') as raw:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as writer:
                    shutil.copyfileobj(reader, writer)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        entries.append({'source': str(path), 'archive': str(dest.relative_to(REPORTS)),
                        'source_sha256': sha(path), 'archive_sha256': sha(dest),
                        'source_bytes': path.stat().st_size, 'archive_bytes': dest.stat().st_size})
    save(target / 'ARCHIVE_MANIFEST.json', {'status': 'COMPLETE', 'source': str(source), 'files': entries})
    print(json.dumps({'relative':relative,'files':len(entries),'bytes':sum(e['archive_bytes'] for e in entries)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('relative')
    archive(parser.parse_args().relative)

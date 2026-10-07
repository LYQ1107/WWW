"""Retain small Phase VI evidence with source and archived SHA bindings."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

from jev_phase6_common import OUT, REPORTS, sha, save, protect_anchor


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

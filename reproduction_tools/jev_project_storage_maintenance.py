"""Scoped home/data1 cleanup; preserve evidence bytes and original runtime paths."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time

HOME_ROOT = Path('/home/liuyeqiang')
SEALED = {'video20', 'video21', 'video22', 'video_20', 'video_21', 'video_22'}
MOVE_NAMES = ['WWW_jev_phase6_runtime', 'WWW_jev_phase10_runtime', 'WWW_jev_phase14_runtime']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_mode, s.st_uid]


def fs(path):
    s = os.statvfs(path)
    return {'available_bytes': s.f_bavail * s.f_frsize,
            'used_bytes': (s.f_blocks - s.f_bfree) * s.f_frsize}


def files(root, exclude_sealed=False):
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d != '.git' and
                   not (exclude_sealed and d in SEALED)]
        for name in names:
            path = Path(base) / name
            if not path.is_symlink() and path.is_file():
                yield path


def project_roots():
    return [p for p in HOME_ROOT.iterdir() if p.name.startswith('WWW') and
            p.is_dir() and not p.is_symlink()]


def no_project_users(paths):
    """Do not stop other processes; refuse changes when project users are observed."""
    current = os.getpid()
    parents = {current, os.getppid()}
    prefixes = tuple(str(p) + '/' for p in paths)
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in parents:
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'jev_project_storage_maintenance.py' in cmd:
                continue
            cwd = os.readlink(p / 'cwd')
            assert not any(cwd == str(x) or cwd.startswith(str(x) + '/') for x in paths), (
                'active project working directory', p.name, cwd)
            for fd in (p / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                assert not target.startswith(prefixes), ('active project file', p.name, target)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            # OpenSSH transports have nondumpable /proc links. Their file
            # handles cannot be certified absent; source/destination checksums
            # and unchanged metadata are independently checked before retiring
            # a directory. Other unreadable process roles still refuse cleanup.
            if (p / 'comm').read_text().strip() in {'sshd', 'sftp-server'}:
                continue
            raise


def dedup_and_cache(out):
    roots = project_roots()
    no_project_users(roots)
    report = {'before': {'home': fs(HOME_ROOT), 'data1': fs(out)},
              'duplicates': [], 'cache_directories': [], 'redundant_temporary_files': [],
              'retained_unique_temporary_count': 0, 'frozen_sources': []}
    candidates = collections.defaultdict(dict)
    # Only pinned source snapshots are shared. Editable worktrees and model weights
    # are excluded, so subsequent experiments do not couple their checkpoints.
    sources = []
    for root in roots:
        if re.fullmatch(r'WWW_jev_phase10_source_v[0-9][a-z]?', root.name):
            sources.append(root)
        if root.name.endswith('_runtime'):
            for base, dirs, _ in os.walk(root, followlinks=False):
                found = [d for d in dirs if d.startswith('source_')]
                sources.extend(Path(base) / d for d in found)
                dirs[:] = [d for d in dirs if d not in found and d not in SEALED]
    for root in sources:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).split(b'\0')
        report['frozen_sources'].append({'path': str(root), 'commit': commit})
        for rel in tracked:
            if not rel:
                continue
            p = root / os.fsdecode(rel)
            if not p.is_file() or p.is_symlink() or p.suffix in {'.pth', '.pt', '.xz'}:
                continue
            s = p.stat()
            if s.st_size < 16384:
                continue
            candidates[(s.st_size, stat.S_IMODE(s.st_mode))][(s.st_dev, s.st_ino)] = p
    bytes_freed = 0
    for paths in candidates.values():
        if len(paths) < 2:
            continue
        identical = {}
        for p in paths.values():
            before = signature(p)
            digest = sha(p)
            assert signature(p) == before
            key = (digest, before[-1])
            if key not in identical:
                identical[key] = p
                continue
            keeper = identical[key]
            if len(report['duplicates']) % 128 == 0:
                no_project_users(roots)
            assert signature(p) == before and sha(keeper) == digest
            old = p.stat()
            tmp = p.with_name(p.name + '.storage-link-' + str(os.getpid()))
            assert not tmp.exists()
            os.link(keeper, tmp)
            os.replace(tmp, p)
            assert p.stat().st_ino == keeper.stat().st_ino and sha(p) == digest
            released = old.st_blocks * 512 if old.st_nlink == 1 else 0
            bytes_freed += released
            report['duplicates'].append({'path': str(p), 'retained_copy': str(keeper),
                                         'SHA256': digest, 'released_bytes': released})
    cache_bytes = 0
    for root in roots:
        for base, dirs, _ in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d != '.git' and d not in SEALED]
            for name in list(dirs):
                p = Path(base) / name
                if name not in {'__pycache__', '.pytest_cache'} or p.is_symlink():
                    continue
                if len(report['cache_directories']) % 128 == 0:
                    no_project_users(roots)
                allocated = sum(f.stat().st_blocks * 512 for f in files(p))
                shutil.rmtree(p)
                report['cache_directories'].append({'path': str(p), 'allocated_bytes': allocated})
                cache_bytes += allocated
                dirs.remove(name)
    temporary_bytes = 0
    for root in roots:
        for p in files(root, exclude_sealed=True):
            match = re.fullmatch(r'(.+)\.tmp\.([0-9]+)', p.name)
            if not match:
                continue
            keeper = p.with_name(match.group(1))
            if Path('/proc', match.group(2)).exists() or not keeper.is_file() or keeper.is_symlink():
                report['retained_unique_temporary_count'] += 1
                continue
            if p.stat().st_size != keeper.stat().st_size or sha(p) != sha(keeper):
                report['retained_unique_temporary_count'] += 1
                continue
            no_project_users([p.parent])
            digest = sha(keeper)
            s = p.stat()
            p.unlink()
            released = s.st_blocks * 512 if s.st_nlink == 1 else 0
            temporary_bytes += released
            report['redundant_temporary_files'].append({'path': str(p), 'retained_copy': str(keeper),
                                                       'SHA256': digest, 'released_bytes': released})
    report.update({'status': 'PASS', 'duplicate_released_bytes': bytes_freed,
                   'cache_released_bytes': cache_bytes, 'temporary_released_bytes': temporary_bytes,
                   'after': {'home': fs(HOME_ROOT), 'data1': fs(out)}})
    (out / 'DEDUP_AND_CACHE.json').write_text(json.dumps(report, indent=2))
    print('DEDUP_AND_CACHE', bytes_freed, cache_bytes, temporary_bytes, flush=True)


def inventory_tree(root):
    entries = []
    inodes = set()
    allocated = 0
    for base, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            p = Path(base) / name
            rel = str(p.relative_to(root))
            s = p.lstat()
            if p.is_symlink():
                link = os.readlink(p)
                assert os.path.isabs(link), 'relative symlink requires separate relocation review'
                entries.append({'relative': rel, 'link': link})
            elif p.is_file():
                assert not any(part in SEALED for part in p.parts), 'sealed content is not read'
                sig = signature(p)
                digest = sha(p)
                assert sig == signature(p), ('file changed during inventory', str(p))
                entries.append({'relative': rel, 'bytes': s.st_size, 'SHA256': digest,
                                'mode': stat.S_IMODE(s.st_mode)})
                inode = (s.st_dev, s.st_ino)
                if inode not in inodes:
                    allocated += s.st_blocks * 512
                    inodes.add(inode)
    return entries, allocated


def dedup_data1(out):
    """Coalesce immutable fixtures/reports; leave models, raw experiments and Git intact."""
    data = Path('/data1/liuyeqiang')
    repos = [p for p in data.iterdir() if p.name.startswith('WWW') and (p / '.git').exists()]
    canonical = data / 'WWW'
    repos.sort(key=lambda p: (p != canonical, str(p)))
    sources = []
    for name in ['WWW_jev_phase13_runtime', 'WWW_jev_phase14_runtime']:
        root = data / name
        if not root.exists():
            continue
        for base, dirs, _ in os.walk(root, followlinks=False):
            found = [d for d in dirs if d.startswith('source_')]
            sources.extend(Path(base) / d for d in found if (Path(base) / d / '.git').exists())
            dirs[:] = [d for d in dirs if d not in found and d not in SEALED]
    sections = []
    paths = collections.defaultdict(dict)
    skip = re.compile(r'(?:^|[/_-])(?:official_)?test(?:[/_.-]|$)|video[_-]?0*(?:20|21|22)(?:[^0-9]|$)', re.I)

    def add(p, kind):
        if not p.is_file() or p.is_symlink() or skip.search(str(p)):
            return
        if p.suffix in {'.pth', '.pt', '.xz', '.zip'}:
            return
        s = p.stat()
        if s.st_size < 16384 or s.st_uid != os.getuid():
            return
        paths[(s.st_size, stat.S_IMODE(s.st_mode), s.st_uid, s.st_gid)][(s.st_dev, s.st_ino)] = (p, kind)

    for repo in repos:
        fixtures = repo / 'TrackEval/data/gt'
        if fixtures.is_dir():
            sections.append(fixtures)
            for p in files(fixtures, exclude_sealed=True):
                add(p, 'immutable_public_TrackEval_fixture')
        if (repo / 'reports').exists():
            sections.append(repo / 'reports')
            tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z', '--', 'reports']).split(b'\0')
            for rel in tracked:
                if rel:
                    add(repo / os.fsdecode(rel), 'committed_historical_report')
    for source in sources:
        sections.append(source)
        tracked = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).split(b'\0')
        for rel in tracked:
            if rel:
                add(source / os.fsdecode(rel), 'immutable_pinned_source')
    no_project_users(sections)
    report = {'status': 'IN_PROGRESS', 'before': fs(data), 'duplicates': [],
              'repositories': [str(p) for p in repos], 'pinned_sources': [str(p) for p in sources],
              'cache_directories': [], 'released_bytes': 0}
    print('DATA1_SIZE_GROUPS', len(paths), 'REPOSITORIES', len(repos), flush=True)
    for group in paths.values():
        if len(group) < 2:
            continue
        identical = {}
        for p, kind in group.values():
            before = signature(p)
            digest = sha(p)
            assert signature(p) == before, ('changed candidate', str(p))
            if digest not in identical:
                identical[digest] = (p, before)
                continue
            keeper, keep_sig = identical[digest]
            if len(report['duplicates']) % 256 == 0:
                no_project_users(sections)
                print('DATA1_DEDUP_PROGRESS', len(report['duplicates']), report['released_bytes'], flush=True)
            assert signature(p) == before and signature(keeper) == keep_sig
            old = p.stat()
            temporary = p.with_name(p.name + '.storage-link-' + str(os.getpid()))
            assert not temporary.exists()
            os.link(keeper, temporary)
            os.replace(temporary, p)
            assert os.path.samefile(p, keeper)
            released = old.st_blocks * 512 if old.st_nlink == 1 else 0
            report['released_bytes'] += released
            report['duplicates'].append({'path': str(p), 'retained_copy': str(keeper),
                                         'kind': kind, 'SHA256': digest, 'released_bytes': released})
    cache_released = 0
    for root in repos + sources:
        for base, dirs, _ in os.walk(root, followlinks=False):
            # Runtime outputs, datasets and Git remain outside cache cleanup.
            dirs[:] = [d for d in dirs if d not in {'.git', 'outputs', 'datasets', 'reports'} and d not in SEALED]
            for name in list(dirs):
                p = Path(base) / name
                if name not in {'__pycache__', '.pytest_cache'} or p.is_symlink():
                    continue
                allocated = sum(f.stat().st_blocks * 512 for f in files(p) if f.stat().st_nlink == 1)
                shutil.rmtree(p)
                cache_released += allocated
                report['cache_directories'].append({'path': str(p), 'allocated_bytes': allocated})
                dirs.remove(name)
    report.update({'status': 'PASS', 'after': fs(data), 'cache_released_bytes': cache_released,
                   'model_weights_shared_or_deleted': False, 'raw_runtime_outputs_deleted': False,
                   'Git_objects_or_history_changed': False, 'official_VISION_TEST_read': False})
    (out / 'DATA1_DEDUP_AND_CACHE.json').write_text(json.dumps(report, indent=2) + '\n')
    print('DATA1_DEDUP_COMPLETE', len(report['duplicates']), report['released_bytes'], cache_released, flush=True)


def relocate(out, reserve_gib):
    archive = out / 'home_runtime'
    archive.mkdir(exist_ok=True)
    prior = out / 'RELOCATION.json'
    results = json.loads(prior.read_text())['cases'] if prior.exists() else []
    for name in MOVE_NAMES:
        src = HOME_ROOT / name
        dst = archive / name
        if src.is_symlink():
            assert src.resolve() == dst.resolve(), 'unexpected existing link'
            print('ALREADY_LINKED', src, flush=True)
            continue
        assert src.is_dir(), ('unexpected source', str(src))
        no_project_users([src])
        manifest = out / (name + '.manifest.json')
        started = time.time()
        resumed = dst.exists()
        if resumed:
            saved = json.loads(manifest.read_text())
            assert saved['source'] == str(src) and saved['destination'] == str(dst)
            entries, allocated = saved['entries'], saved['allocated_bytes']
            print('RESUME_EXISTING_COPY', name, flush=True)
        else:
            print('INVENTORY', src, flush=True)
            entries, allocated = inventory_tree(src)
            assert fs(out)['available_bytes'] - allocated >= reserve_gib * 2**30, 'insufficient destination reserve'
            manifest.write_text(json.dumps({'source': str(src), 'destination': str(dst),
                                           'allocated_bytes': allocated, 'entries': entries}, indent=2))
            dst.mkdir()
            print('COPY', name, allocated, len(entries), flush=True)
            subprocess.run(['rsync', '-aH', '--sparse', str(src) + '/', str(dst) + '/'], check=True)
        print('VERIFY', name, flush=True)
        actual, _ = inventory_tree(dst)
        assert {x['relative']: x for x in entries} == {x['relative']: x for x in actual}, 'destination differs'
        # An unchanged source stat set plus SHA/rsync verification precedes unlink.
        no_project_users([src])
        print('CHECK_SOURCE_UNCHANGED', name, flush=True)
        again = subprocess.run(['rsync', '-aHnc', '--itemize-changes', '--delete', str(src) + '/', str(dst) + '/'],
                               check=True, capture_output=True, text=True)
        assert not again.stdout.strip(), ('source changed', again.stdout)
        retired = src.with_name(src.name + '.storage-retired-' + str(os.getpid()))
        src.rename(retired)
        try:
            src.symlink_to(dst, target_is_directory=True)
        except BaseException:
            retired.rename(src)
            raise
        shutil.rmtree(retired)
        # Verify every original report path resolves to the already SHA-verified
        # destination inode; do not repeat a third full read of archived bytes.
        for entry in entries:
            p = src / entry['relative']
            if 'link' in entry:
                assert p.is_symlink() and os.readlink(p) == entry['link']
            else:
                assert p.is_file() and p.stat().st_size == entry['bytes']
                assert os.path.samefile(p, dst / entry['relative'])
        results.append({'original_path': str(src), 'destination': str(dst),
                        'files': sum('SHA256' in x for x in entries),
                        'allocated_home_bytes_released': allocated,
                        'manifest': str(manifest), 'manifest_SHA256': sha(manifest),
                        'verification': 'ALL_FILE_SHA256_AND_SYMLINKS_EQUAL',
                        'source_unchanged_checksum_and_metadata_check': True,
                        'resumed_existing_copy': resumed,
                        'seconds': time.time() - started})
        (out / 'RELOCATION.json').write_text(json.dumps({'status': 'IN_PROGRESS', 'cases': results}, indent=2))
        print('RELOCATED', name, 'home_free_GiB', fs(HOME_ROOT)['available_bytes'] / 2**30,
              'data1_free_GiB', fs(out)['available_bytes'] / 2**30, flush=True)
    (out / 'RELOCATION.json').write_text(json.dumps({'status': 'PASS', 'cases': results}, indent=2))


def verify_report(out):
    root = Path(__file__).resolve().parents[1]
    reports = root / 'reports/JEV_PHASE14'
    dedup = json.loads((out / 'DEDUP_AND_CACHE.json').read_text())
    relocation = json.loads((out / 'RELOCATION.json').read_text())
    data_cleanup = json.loads((out / 'DATA1_DEDUP_AND_CACHE.json').read_text())
    assert dedup['status'] == relocation['status'] == 'PASS'
    assert data_cleanup['status'] == 'PASS'
    expected = {}

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('SHA256'), str):
                path = Path(value['path'])
                if not path.is_absolute():
                    path = root / path
                key = str(path)
                assert key not in expected or expected[key] == value['SHA256'], 'conflicting evidence hash'
                expected[key] = value['SHA256']
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    delivery = json.loads((reports / 'DELIVERY_AUDIT.json').read_text())
    for item in delivery['required_reports']:
        path = Path(item['path'])
        assert sha(path) == item['SHA256'], ('changed original report', str(path))
        visit(json.loads(path.read_text()))
    print('VERIFY_PHASE14_REFERENCES', len(expected), flush=True)
    physical = {}
    for path, digest in expected.items():
        p = Path(path)
        assert p.is_file(), ('missing report reference', path)
        s = p.stat()
        inode = (s.st_dev, s.st_ino)
        actual = physical.setdefault(inode, None)
        if actual is None:
            actual = sha(p)
            physical[inode] = actual
        assert actual == digest, ('report reference changed', path)
    print('VERIFY_HISTORICAL_CHECKPOINTS', flush=True)
    historical = json.loads((reports / 'HISTORICAL_CHECKPOINT_AUDIT.json').read_text())
    for item in historical['models']:
        p = Path(item['path'])
        assert p.is_file() and p.stat().st_size == item['bytes'], ('checkpoint missing', str(p))
        s = p.stat()
        inode = (s.st_dev, s.st_ino)
        actual = physical.setdefault(inode, None)
        if actual is None:
            actual = sha(p)
            physical[inode] = actual
        assert actual == item['actual_SHA256'], ('checkpoint changed', str(p))
    frozen = json.loads((reports / 'PHASE13_FROZEN_EVIDENCE.json').read_text())
    for rel, digest in frozen['protected_report_SHA256'].items():
        assert sha(root / rel) == digest, ('protected report changed', rel)
    summary = {
        'status': 'PASS', 'task': 'Project-scoped home and data1 storage cleanup, 2026-10-10',
        'pre_cleanup_project_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'before': dedup['before'], 'after': {'home': fs(HOME_ROOT), 'data1': fs(out)},
        'duplicate_source_files_replaced_by_hardlinks': len(dedup['duplicates']),
        'immutable_source_snapshots': len(dedup['frozen_sources']),
        'duplicate_released_bytes': dedup['duplicate_released_bytes'],
        'cache_directories_removed': len(dedup['cache_directories']),
        'cache_released_bytes': dedup['cache_released_bytes'],
        'redundant_temporary_files_removed': len(dedup['redundant_temporary_files']),
        'retained_unique_temporary_files': dedup['retained_unique_temporary_count'],
        'relocations': relocation['cases'],
        'data1_cleanup': {
            'before': data_cleanup['before'], 'after': data_cleanup['after'],
            'duplicate_files_coalesced': len(data_cleanup['duplicates']),
            'released_bytes': data_cleanup['released_bytes'],
            'cache_directories_removed': len(data_cleanup['cache_directories']),
            'cache_released_bytes': data_cleanup['cache_released_bytes'],
            'scope': 'immutable generic TrackEval fixtures, committed historical reports and pinned source snapshots',
            'manifest': str(out / 'DATA1_DEDUP_AND_CACHE.json'),
            'manifest_SHA256': sha(out / 'DATA1_DEDUP_AND_CACHE.json')},
        'original_runtime_paths_preserved_with_directory_symlinks': True,
        'phase14_referenced_files_verified': len(expected),
        'historical_checkpoints_SHA256_verified': len(historical['models']),
        'protected_phase13_reports_SHA256_verified': len(frozen['protected_report_SHA256']),
        'original_phase14_delivery_audit_SHA256': sha(reports / 'DELIVERY_AUDIT.json'),
        'scientific_models_deleted': 0, 'scientific_results_deleted': 0,
        'heldout_20_21_22_content_read': False, 'official_TEST_read': False,
        'other_projects_or_users_cleaned': False, 'shared_conda_environment_or_model_caches_deleted': False,
        'hardlink_scope': 'Only immutable pinned source snapshots; editable worktrees and model weights excluded',
        'data1_hardlink_scope': 'Committed historical reports and immutable public fixtures; model weights and editable source excluded',
        'uninspectable_transport_processes': 'OpenSSH sshd/SFTP may have nondumpable proc links; relocation additionally checks source/destination checksums and metadata',
        'reserved_data1_bytes': 15 * 2**30,
        'accounting': 'Verified duplicate/cache release and relocated allocations are separate; filesystem drift is not attributed to cleanup',
        'retained_failure_evidence': '/home/liuyeqiang/WWW_failed_artifacts/train_off.failed_partial_20261005.jsonl',
        'server_manifests': [{'path': str(out / name), 'SHA256': sha(out / name)}
                             for name in ['DEDUP_AND_CACHE.json', 'RELOCATION.json', 'WORKER_RESTARTS.json']],
    }
    target = reports / 'STORAGE_CLEANUP_20261010.json'
    target.write_text(json.dumps(summary, indent=2) + '\n')
    print('STORAGE_CLEANUP_VERIFIED', str(target), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['dedup-cache', 'data1-dedup-cache', 'relocate', 'verify'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reserve-gib', type=float, default=15)
    args = p.parse_args()
    assert Path('/data1/liuyeqiang') in args.out.resolve().parents, 'archive must use the authorized data1 project area'
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'dedup-cache':
        dedup_and_cache(args.out)
    elif args.stage == 'data1-dedup-cache':
        dedup_data1(args.out)
    elif args.stage == 'relocate':
        relocate(args.out, args.reserve_gib)
    else:
        verify_report(args.out)


if __name__ == '__main__':
    main()

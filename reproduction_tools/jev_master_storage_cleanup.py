"""Lossless archival of an explicit, inactive TRAIN-log allowlist; no data download.

Archives stay beside their sources. Legacy raw-path readers must restore first.
No source removal is allowed before full decompressed SHA256 verification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time

ZSTD = '/home/liuyeqiang/anaconda3/bin/zstd'
CHUNK = 4 * 1024 * 1024
RESERVE = 5 * 2**30
DEFAULT_ROOT = Path('/home/liuyeqiang/WWW_jev_master_audit_runtime/storage_cleanup_20261012')
ALLOWLIST = {
    '/home/liuyeqiang/old_proxy_trace_quarantine/train_off_gpu2_isolated.jsonl',
    '/home/liuyeqiang/WWW_failed_artifacts/train_off.failed_partial_20261005.jsonl',
    '/data1/liuyeqiang/WWW/outputs/research_v2/model_4500/traces/train_off_full_v2.failed_20261004T165344Z.jsonl',
    '/data1/liuyeqiang/WWW/outputs/research_v2/model_4500/traces/train_off_full_v2.jsonl',
    '/data1/liuyeqiang/WWW/outputs/research_final_v2/off/traces/train_off.jsonl',
    '/data1/liuyeqiang/WWW/outputs/research_final_v2/off/traces/train_tail_off.jsonl',
}


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def fingerprint(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_nlink != 1:
        raise RuntimeError('Source must be an owned, single-link regular file')
    return dict(bytes=s.st_size, allocated_bytes=s.st_blocks * 512,
                inode=s.st_ino, device=s.st_dev, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                mode=stat.S_IMODE(s.st_mode), atime_ns=s.st_atime_ns)


def unchanged(path, expected):
    current = fingerprint(path)
    for key in ('bytes', 'inode', 'device', 'mtime_ns', 'ctime_ns'):
        if key not in expected:
            continue
        if current[key] != expected[key]:
            raise RuntimeError('Source changed: ' + key)


def no_process_reference(path):
    """Check inspectable same-user references; disclose inaccessible SSH services.

    Linux may deny ptrace inspection of privilege-separated SSH/SFTP services.
    These exceptions are logged, not described as an exhaustive no-open proof.
    Source inode/size/mtime/ctime stability remains mandatory before removal.
    """
    target = str(path)
    service_exceptions = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            status = (proc / 'status').read_text()
            if any(line.startswith('State:\tZ') for line in status.splitlines()):
                continue  # Zombies have released all descriptors and mappings.
            for fd in (proc / 'fd').iterdir():
                try:
                    ref = os.readlink(fd).removesuffix(' (deleted)')
                except FileNotFoundError:
                    continue
                if ref == target:
                    raise RuntimeError('Source open by PID ' + proc.name)
            maps = (proc / 'maps').read_text()
            if any(line.endswith(' ' + target) for line in maps.splitlines()):
                raise RuntimeError('Source mapped by PID ' + proc.name)
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            if os.fsencode(target) in argv:
                raise RuntimeError('Source argument of PID ' + proc.name)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as e:
            name = next((line.split('\t', 1)[1] for line in status.splitlines()
                         if line.startswith('Name:\t')), '')
            if name not in ('sshd', 'sftp-server'):
                raise RuntimeError('Cannot inspect non-SSH same-user process') from e
            service_exceptions.append(dict(pid=int(proc.name), name=name,
                                           limitation='fd/maps inaccessible'))
    return service_exceptions


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def decompress_verify(archive, expected_sha, expected_bytes, destination=None):
    h = hashlib.sha256()
    count = 0
    with subprocess.Popen([ZSTD, '-q', '-d', '-c', str(archive)],
                          stdout=subprocess.PIPE) as proc:
        for block in iter(lambda: proc.stdout.read(CHUNK), b''):
            h.update(block)
            count += len(block)
            if destination is not None:
                destination.write(block)
        if proc.wait() != 0:
            raise RuntimeError('Decompression failed')
    if h.hexdigest() != expected_sha or count != expected_bytes:
        raise RuntimeError('Round-trip SHA256/byte count mismatch')
    return dict(sha256=h.hexdigest(), bytes=count)


def archive_one(path, expected, ledger, ledger_path):
    archive = Path(str(path) + '.zst')
    partial = Path(str(archive) + '.partial')
    if archive.exists() or partial.exists():
        raise RuntimeError('Existing archive/partial requires explicit recovery')
    unchanged(path, expected)
    process_exceptions = no_process_reference(path)
    before = fingerprint(path)
    started = time.time()
    raw_hash = hashlib.sha256()
    count = 0
    last_progress = 0.0
    print('ARCHIVE_START ' + str(path), flush=True)
    with partial.open('xb') as output:
        with subprocess.Popen([ZSTD, '-q', '-1', '-T2', '-c'],
                              stdin=subprocess.PIPE, stdout=output) as proc:
            try:
                with path.open('rb') as source:
                    for block in iter(lambda: source.read(CHUNK), b''):
                        if count % (64 * 2**20) == 0:
                            if shutil.disk_usage(path.parent).free < RESERVE:
                                raise RuntimeError('Free-space reserve reached; source retained')
                        raw_hash.update(block)
                        proc.stdin.write(block)
                        count += len(block)
                        if time.time() - last_progress > 10:
                            print(json.dumps(dict(progress_path=str(path),
                                                  bytes=count, total=before['bytes'])), flush=True)
                            last_progress = time.time()
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError('Compression failed; source retained')
            except BaseException:
                proc.kill()
                proc.wait()
                raise
        output.flush()
        os.fsync(output.fileno())
    if count != before['bytes']:
        raise RuntimeError('Source size changed during compression')
    print('ROUNDTRIP_VERIFY ' + str(path), flush=True)
    verification = decompress_verify(partial, raw_hash.hexdigest(), count)
    unchanged(path, before)
    no_process_reference(path)
    compressed_sha = sha_file(partial)
    partial.rename(archive)
    archived = archive.stat()
    record = dict(path=str(path), archive_path=str(archive), original=before,
                  raw_sha256=raw_hash.hexdigest(), archive_sha256=compressed_sha,
                  archive_bytes=archived.st_size, archive_allocated_bytes=archived.st_blocks*512,
                  roundtrip=verification, source_unlinked=False,
                  process_inspection_exceptions=process_exceptions,
                  status='VERIFIED_ARCHIVE_SOURCE_RETAINED')
    ledger['files'].append(record)
    save(ledger_path, ledger)
    # Repeat the guards immediately before releasing the raw copy.
    unchanged(path, before)
    no_process_reference(path)
    path.unlink()
    record.update(source_unlinked=True, status='ARCHIVED_AND_RAW_COPY_RELEASED',
                  freed_allocated_bytes=before['allocated_bytes']-archived.st_blocks*512,
                  elapsed_seconds=round(time.time()-started, 3))
    save(ledger_path, ledger)
    print(json.dumps(record), flush=True)


def restore(raw_path, ledger_path):
    ledger = json.loads(ledger_path.read_text())
    record = next(r for r in ledger['files'] if r['path'] == str(raw_path))
    archive = Path(record['archive_path'])
    tmp = Path(str(raw_path) + '.restore.tmp')
    if raw_path.exists() or tmp.exists():
        raise RuntimeError('Restore refuses to overwrite existing files')
    if sha_file(archive) != record['archive_sha256']:
        raise RuntimeError('Compressed archive SHA256 changed')
    if shutil.disk_usage(raw_path.parent).free < record['original']['bytes'] + RESERVE:
        raise RuntimeError('Insufficient free space for restoration plus reserve')
    with tmp.open('xb') as f:
        decompress_verify(archive, record['raw_sha256'], record['original']['bytes'], f)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, record['original']['mode'])
    os.utime(tmp, ns=(record['original']['atime_ns'], record['original']['mtime_ns']))
    # Link creation is exclusive: an intervening raw-path creation is never overwritten.
    os.link(tmp, raw_path)
    tmp.unlink()
    print('RESTORED ' + str(raw_path), flush=True)


def selftest():
    with tempfile.TemporaryDirectory(prefix='jev_archive_test_') as d:
        root = Path(d)
        source = root/'train_fixture.jsonl'
        source.write_bytes((b'{"id":12,"negative_evidence":true}\n' * 1500) + b'\x00\xff')
        expected = fingerprint(source)
        ledger = dict(files=[])
        ledger_path = root/'ledger.json'
        archive_one(source, expected, ledger, ledger_path)
        assert not source.exists()
        restore(source, ledger_path)
        assert sha_file(source) == ledger['files'][0]['raw_sha256']
        rejected = []
        for name, action in [
            ('restore_overwrite', lambda: restore(source, ledger_path)),
            ('existing_archive', lambda: archive_one(source, fingerprint(source), ledger, ledger_path)),
            ('wrong_hash', lambda: decompress_verify(Path(str(source)+'.zst'), '0'*64, expected['bytes'])),
        ]:
            try:
                action()
            except RuntimeError:
                rejected.append(name)
            else:
                raise AssertionError('Guard did not reject ' + name)
        source.write_bytes(b'changed')
        try:
            unchanged(source, expected)
        except RuntimeError:
            rejected.append('changed_source')
        else:
            raise AssertionError('Changed source was accepted')
        # Another live process holding the source must block archival.
        p = subprocess.Popen(['/home/liuyeqiang/anaconda3/envs/GMT/bin/python', '-c',
                              'import sys,time; f=open(sys.argv[1],"rb"); time.sleep(30)', str(source)])
        try:
            time.sleep(0.2)
            try:
                no_process_reference(source)
            except RuntimeError:
                rejected.append('live_process_reference')
            else:
                raise AssertionError('Live reference accepted')
        finally:
            p.terminate()
            p.wait()
        print(json.dumps(dict(status='PASS', lossless_restore=True, guards=rejected)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--restore', type=Path)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    ledger_path = args.root/'ARCHIVE_LEDGER.json'
    if args.restore:
        if str(args.restore) not in ALLOWLIST:
            raise RuntimeError('Restore path outside frozen allowlist')
        restore(args.restore, ledger_path)
        return
    plan = json.loads((args.root/'PLAN.json').read_text())
    if {f['path'] for f in plan['files']} != ALLOWLIST:
        raise RuntimeError('Plan does not match frozen six-file TRAIN allowlist')
    for entry in plan['files']:
        if entry['archive_path'] != entry['path']+'.zst':
            raise RuntimeError('Archive destination changed')
        unchanged(Path(entry['path']), entry)
        if time.time_ns()-entry['mtime_ns'] < 24*3600*10**9:
            raise RuntimeError('Source is not old enough to archive')
        no_process_reference(Path(entry['path']))
    if not args.execute:
        print(json.dumps(dict(status='DRY_RUN_PASS', files=len(ALLOWLIST))))
        return
    if ledger_path.exists():
        raise RuntimeError('Existing ledger requires explicit recovery; no automatic rerun')
    ledger = dict(status='RUNNING', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  files=[], zstd_level=1, cpu_threads=2, gpu_use=False,
                  semantic_GT_or_trace_evaluation=False)
    save(ledger_path, ledger)
    try:
        for entry in plan['files']:
            archive_one(Path(entry['path']), entry, ledger, ledger_path)
        ledger['status'] = 'COMPLETE'
    except BaseException as exc:
        ledger.update(status='STOPPED_WITH_SOURCES_RETAINED_FOR_UNFINISHED_FILES', error=str(exc))
        raise
    finally:
        ledger['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save(ledger_path, ledger)


if __name__ == '__main__':
    main()

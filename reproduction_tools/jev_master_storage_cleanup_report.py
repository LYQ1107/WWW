"""Publish compact cleanup evidence after all six archives have been verified."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('/home/liuyeqiang/WWW_jev_master_audit_runtime/storage_cleanup_20261012')
REPORTS = ROOT/'reports/JEV_MASTER'


def read(name):
    return json.loads((RUN/name).read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ledger = read('ARCHIVE_LEDGER.json')
    assert ledger['status'] == 'COMPLETE' and len(ledger['files']) == 6
    verified = []
    for record in ledger['files']:
        archive = Path(record['archive_path'])
        assert record['source_unlinked'] and not Path(record['path']).exists()
        assert archive.stat().st_size == record['archive_bytes']
        assert sha(archive) == record['archive_sha256']
        assert record['roundtrip']['sha256'] == record['raw_sha256']
        assert record['roundtrip']['bytes'] == record['original']['bytes']
        verified.append(str(archive))
    cache = read('CACHE_CLEANUP.json')
    frozen = read('FROZEN_ARTIFACT_INTEGRITY.json')
    weights = read('CORE_WEIGHT_INTEGRITY.json')
    assert frozen['status'] == weights['status'] == 'PASS'
    selftest = json.loads((RUN/'SELFTEST.log').read_text().splitlines()[-1])
    assert selftest['status'] == 'PASS'
    after = {p: dict(zip(('total', 'used', 'free'), shutil.disk_usage(p)))
             for p in ('/home/liuyeqiang', '/data1/liuyeqiang')}
    (RUN/'FILESYSTEM_AFTER.json').write_text(json.dumps(after, indent=2)+'\n')
    released = {p: sum(r['freed_allocated_bytes'] for r in ledger['files']
                       if r['path'].startswith(p+'/')) for p in after}
    released['/home/liuyeqiang'] += cache['freed_allocated_file_bytes']
    result = dict(
        status='COMPLETE_VERIFIED_LOSSLESS_STORAGE_CLEANUP',
        generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        execution_code_commit='c6a69ce080c7db59eea136ac28ab9fed145e5fba',
        execution_archive_code_sha256=sha(ROOT/'reproduction_tools/jev_master_storage_cleanup.py'),
        user_authorization='Clean unused project storage on home and data1',
        archive_ledger=ledger, cache_cleanup=cache,
        selftest=selftest, frozen_publication_integrity=frozen, core_weight_integrity=weights,
        filesystem_initial=read('FILESYSTEM_BEFORE.json'),
        filesystem_immediately_before_execution=read('EXECUTION_FILESYSTEM_BEFORE.json'),
        filesystem_after=after,
        attributable_released_allocated_bytes=released,
        attributable_total_released_allocated_bytes=sum(released.values()),
        measurement_note='Allocated original blocks minus retained archive blocks; cache last-link file blocks only. Filesystem free-space drift also includes other processes and is not attributed to this cleanup.',
        archives_verified_present_after_raw_unlink=verified,
        preservation=dict(models_and_checkpoints=True, datasets_and_GT=True,
                          current_perception_and_native_caches=True,
                          official_TEST_and_sealed20_21_22=True,
                          all_scientific_reports_and_original_negative_results=True),
        no_new_dataset_downloads=True, no_GPU_use=True, no_scientific_reexecution=True,
        master_scientific_goal_status_unchanged=True,
        process_inspection_limit='Same-user accessible fd/maps/exact argv checked. Privilege-separated SSH/SFTP services deny inspection and are disclosed in each record. No exhaustive no-open-file claim; raw inode/size/mtime/ctime stability was mandatory.',
        restore_note='Old raw-path readers must restore that specific file first. Archives contain byte-identical logs; no compressed bytes are silently substituted at raw JSONL paths.',
        cache_commands=['conda clean --tarballs --index-cache --yes --json',
                        'Remove only owned, non-symlink ~/.cache/pip/{http,http-v2,selfcheck}; no wheel/model cache deletion'],
        no_binary_upload=True,
    )
    result_path = REPORTS/'STORAGE_CLEANUP_20261012.json'
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    rows = '\n'.join('| `%s` | %.3f | %.3f | %.3f |' % (
        r['path'], r['original']['allocated_bytes']/2**30,
        r['archive_allocated_bytes']/2**30, r['freed_allocated_bytes']/2**30)
        for r in ledger['files'])
    report = f'''# WWW / JEV 项目存储清理：2026-10-12

已完成 home 与 data1 的项目存储清理，按文件实际占用块统计共释放 **{sum(released.values())/2**30:.3f} GiB**。六份旧 TRAIN JSONL 日志无损归档，原始失败和负面证据仍保留在压缩包内；另清理可重新下载的安装缓存。

| 文件系统 | 本次可归因释放（GiB） | 完成时可用（GiB） |
| --- | ---: | ---: |
| home（根分区） | {released['/home/liuyeqiang']/2**30:.3f} | {after['/home/liuyeqiang']['free']/2**30:.3f} |
| data1 | {released['/data1/liuyeqiang']/2**30:.3f} | {after['/data1/liuyeqiang']['free']/2**30:.3f} |

`df` 可用空间会随其他进程读写变化，不能把整个分区的变化都算作本次释放。JSON 同时保存开始盘点、执行前、执行后三个快照；释放量采用原文件占用减归档占用，缓存仅计被删除且无其他硬链接的文件块，未计小量目录块。

## 归档记录

| 原文件 | 原占用（GiB） | 保留归档（GiB） | 释放（GiB） |
| --- | ---: | ---: | ---: |
{rows}

每个归档位于原路径加 `.zst`。使用现有 Zstd 1.4.9、level 1、两条 CPU 工作线程，未使用 GPU。流式计算原文件 SHA256，再完整解压计算 SHA256 和字节数；结果完全相同后检查 inode、大小、mtime、ctime 未变，再移除原始未压缩副本。删除后的六个归档也已再次确认存在且压缩包 SHA256 匹配。完整原始/压缩 SHA 和逐文件时间见 [清理记录](../reports/JEV_MASTER/STORAGE_CLEANUP_20261012.json)。

可读取的同用户进程 fd、映射和精确路径参数均经过占用检查。SSH/SFTP 服务因 Linux 权限无法检查 fd/maps，该限制已逐项记录，未声称检查覆盖所有打开的文件。原日志均超过 24 小时未修改，压缩与校验全过程要求源文件元数据保持不变。

## 缓存与保留范围

Conda 仅清 tarballs 和 index-cache，pip 仅清 HTTP/selfcheck 缓存，释放 {cache['freed_allocated_file_bytes']/2**30:.3f} GiB。没有删除 Conda 环境、已安装包目录、HF 模型、MATLAB 安装目录、原始视频/标注、checkpoint、当前 perception/native 缓存、官方 TEST 或 sealed 20/21/22。其他项目和其他用户文件未处理。重复 checkpoint 的预检显示重复路径已指向同一 inode，未再重复计算或删除“副本”。

Phase XVII–XXI 的 {frozen['checked_files']} 个已发布代码/报告文件 SHA256 全部匹配；核心 GMT、两份 Qwen 权重分片、QA adapter 共四个权重文件 SHA256 也全部匹配。未宣称重新校验所有历史权重。未重跑科学实验，MASTER GOAL 的证据门槛与状态保持不变。

## 还原方法

旧程序按原始 `.jsonl` 路径读取时，需先还原对应文件。工具不会将压缩内容伪装成原始 JSONL。

```bash
/home/liuyeqiang/anaconda3/envs/GMT/bin/python \\
  /home/liuyeqiang/WWW_jev_master_audit/reproduction_tools/jev_master_storage_cleanup.py \\
  --restore /完整/原始/train日志.jsonl
```

`--restore` 后应填写上表中的一个完整原路径。默认读取本机 `{RUN}/ARCHIVE_LEDGER.json`；也可用 `--root` 指定包含同名台账的目录。还原会验证压缩包 SHA、解压 SHA 和字节数，拒绝覆盖已有文件，并要求原文件大小加 5 GiB 的剩余空间。压缩包继续保留。

自测覆盖完整还原、拒绝覆盖、已有归档、错误 SHA、源文件变动、活跃进程占用。六份实际归档均通过完整解压校验。GitHub 只上传代码、报告、紧凑 JSON 与 SHA 清单，归档包留在服务器。
'''
    doc = ROOT/'docs/WWW_JEV_STORAGE_CLEANUP_20261012.md'
    doc.write_text(report)
    files = [result_path, doc, REPORTS/'STORAGE_CLEANUP_20261012_PLAN.json',
             ROOT/'reproduction_tools/jev_master_storage_cleanup.py', Path(__file__).resolve()]
    manifest = dict(status='VERIFIED_COMPACT_STORAGE_CLEANUP_PUBLICATION', no_binary_upload=True,
                    files={str(p.relative_to(ROOT)): sha(p) for p in files})
    (REPORTS/'STORAGE_CLEANUP_20261012_SHA_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status=result['status'], freed_GiB=sum(released.values())/2**30,
                         filesystems={p: dict(freed_GiB=released[p]/2**30,
                                             available_GiB=after[p]['free']/2**30) for p in after})))


if __name__ == '__main__':
    main()

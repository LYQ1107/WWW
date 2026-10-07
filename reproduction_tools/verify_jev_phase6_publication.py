"""Verify archived evidence bytes and bounded scientific report bindings."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from jev_phase6_common import OUT, REPORTS, ROOT, sha, save, protect_anchor
from archive_jev_phase6 import projection


def read(path):
    return json.loads(Path(path).read_text())


def digest_stream(reader, digest):
    while True:
        data = reader.read(4 * 1024 * 1024)
        if not data:
            break
        digest.update(data)


def main():
    protect_anchor()
    required = ['FANCY_GATING_AUDIT', 'REASSOCIATE_EVENT_AUDIT',
                'MEMORY_EVENT_DRIVEN_PROTOCOL', 'MEMORY_BASELINE_COMPARISON',
                'REACTIVATION_CANONICAL_DATA_AUDIT', 'REACTIVATION_ACTION_SPACE_AUDIT',
                'CLOSED_LOOP_STATE_AUGMENTATION', 'TYPED_JEV_ARCHITECTURE',
                'UNIFIED_LIFECYCLE_ABLATION', 'HELDOUT_PROTOCOL', 'HELDOUT_RESULTS',
                'FINAL_PHASE6_GO_NO_GO']
    reports = {}
    for name in required:
        path = REPORTS / (name + '.json')
        data = read(path)
        if name != 'HELDOUT_PROTOCOL' and not data.get('what_did_we_learn'):
            raise AssertionError(f'Missing learning statement: {name}')
        reports[name] = sha(path)
    final = read(REPORTS / 'FINAL_PHASE6_GO_NO_GO.json')
    assert read(REPORTS / 'HELDOUT_RESULTS.json')['heldout_protocol_sha256'] == reports['HELDOUT_PROTOCOL']
    assert not final['PHASE6_GO'] and not final['FULL24_AUTHORIZED']
    assert not final['OFFICIAL_TEST_AUTHORIZED']
    assert final['stop_conditions']['second_learned_MEMORY_removed']
    assert final['stop_conditions']['third_learned_REACT_disabled']
    fit = read(OUT / 'standalone_heads_native/REACT/result.json')
    for source, expected in fit['eligibility']['source_sha256'].items():
        assert sha(Path(source)) == expected, source
    labels = sorted((OUT / 'lifecycle_native').glob('video*/labels/*.json'))
    assert len(labels) == 68
    for label in labels:
        data = read(label)
        assert data['control_prefix_rng_and_native_bank_exact'], label
        assert not data['GT_or_future_inputs'], label
        assert len(data['branches']) == 3, label
    paths = sorted((REPORTS / 'evidence').rglob('ARCHIVE_MANIFEST.json'))
    total_files = 0
    total_bytes = 0
    source_count = 0
    projection_sources = []
    manifests = {}
    for manifest_path in paths:
        manifest = read(manifest_path)
        assert manifest['status'] == 'COMPLETE', manifest_path
        groups = {}
        for entry in manifest['files']:
            archive = REPORTS / entry['archive']
            assert sha(archive) == entry['archive_sha256'], archive
            assert archive.stat().st_size == entry['archive_bytes'], archive
            assert archive.stat().st_size < 90 * 1024 * 1024, archive
            groups.setdefault(entry['source'], []).append(entry)
            total_files += 1
            total_bytes += archive.stat().st_size
        for source_name, entries in groups.items():
            source = Path(source_name)
            expected = entries[0]['source_sha256']
            assert all(e['source_sha256'] == expected for e in entries)
            assert sha(source) == expected, source
            source_count += 1
            if 'archive_format' in entries[0]:
                assert len(entries) == 1
                entry = entries[0]
                with gzip.open(REPORTS / entry['archive'], 'rt') as projected:
                    if source.suffix == '.jsonl':
                        with source.open() as original:
                            count = 0
                            for line in original:
                                archived_line = projected.readline()
                                assert archived_line, source
                                assert projection(json.loads(line)) == json.loads(archived_line), source
                                count += 1
                    else:
                        rows = read(source)
                        count = 0
                        for row in rows:
                            archived_line = projected.readline()
                            assert archived_line, source
                            assert projection(row) == json.loads(archived_line), source
                            count += 1
                        del rows
                    assert not projected.readline(), source
                    assert count == entry['records'], source
                projection_sources.append({'source': source_name, 'records': count,
                                           'format': entry['archive_format']})
                continue
            digest = hashlib.sha256()
            offset = 0
            next_line = 1
            for entry in sorted(entries, key=lambda e: e['archive']):
                archive = REPORTS / entry['archive']
                if 'source_byte_bounds' in entry:
                    assert entry['source_byte_bounds'][0] == offset, archive
                    offset = entry['source_byte_bounds'][1]
                if 'jsonl_lines' in entry:
                    assert entry['jsonl_lines'][0] == next_line, archive
                    next_line = entry['jsonl_lines'][1] + 1
                opener = gzip.open if archive.suffix == '.gz' else open
                with opener(archive, 'rb') as reader:
                    digest_stream(reader, digest)
            assert digest.hexdigest() == expected, source
            if offset:
                assert offset == source.stat().st_size, source
        manifests[str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
    phase5 = '/data1/liuyeqiang/WWW_jev_phase5'
    assert subprocess.check_output(['git', '-C', phase5, 'rev-parse', 'HEAD'], text=True).strip() == 'a37083dac0a23eb3846cb252eeb975982aaaabc9'
    assert not subprocess.check_output(['git', '-C', phase5, 'status', '--porcelain'], text=True).strip()
    protect_anchor()
    result = {'status': 'PASS', 'required_reports': reports,
              'archive_manifests': manifests, 'archive_files_verified': total_files,
              'archive_bytes_verified': total_bytes, 'source_files_verified': source_count,
              'lossless_decompressed_source_bindings': 'PASS',
              'diagnostic_projections': projection_sources,
              'projection_complete_controller_inputs_and_actions_checked': True,
              'REACT_training_label_source_bindings': 'PASS',
              'canonical_counterfactual_labels_verified': len(labels),
              'live_counterfactual_branches_verified': len(labels) * 3,
              'B2_and_PhaseV_preserved': True, 'full24_authorized': False,
              'official_test_read': False,
              'scope': 'local archived evidence integrity; remote Git publication is verified separately after final push',
              'what_did_we_learn': 'Archived predictions, targets, checkpoints and evaluator outputs match their declared sources; diagnostic candidate-array projections are explicitly distinguished from lossless files and preserve every controller input/action record.'}
    save(REPORTS / 'PUBLICATION_INTEGRITY_AUDIT.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'archive_files_verified',
                                           'archive_bytes_verified', 'source_files_verified']}))


if __name__ == '__main__':
    main()

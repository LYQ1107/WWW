"""Recorded source consistency, separate from another empirical parity run."""
import ast
from common import *


ENTRY_FILES = {'geo_tools/common.py', 'geo_tools/run_baseline.py', 'geo_tools/run_online.py'}
RETRIEVAL_ONLY = 'gtr/geometry/isolated_stream.py'


def main():
    strict = sorted(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    rows, artifacts, current = [], {}, {}
    def digest(path):
        key = str(path)
        if key not in current:
            current[key] = sha256(path)
        return current[key]
    for name in ('OFF_post_cache_trace_strict_v1_parity.json', 'ZERO_post_cache_trace_strict_v1_parity.json'):
        audit_path = ROOT/'results'/name
        audit = json.loads(audit_path.read_text())
        if audit['status'] != 'PASS' or not audit['full_strict_suite'] or sorted(r['video_id'] for r in audit['rows']) != strict:
            raise ValueError('complete original parity audit required')
        artifacts[str(audit_path)] = digest(audit_path)
        for row in audit['rows']:
            if not row['exact_serialized_predictions']:
                raise ValueError('original parity is not exact')
            paths = [Path(value) for value in row['prediction_paths']]
            manifests = []
            for index, path in enumerate(paths):
                manifest = path.with_name(path.stem+'_manifest.json')
                if not manifest.exists():
                    manifest = path.parent/'manifest.json'
                if digest(path) != row['prediction_sha256'][index] or digest(manifest) != row['run_manifest_sha256'][index]:
                    raise ValueError('changed original prediction/manifest')
                artifacts[str(path)] = digest(path)
                manifests.append(json.loads(manifest.read_text()))
                artifacts[str(manifest)] = digest(manifest)
            run = manifests[1]
            options = run['geo_options']
            if name.startswith('OFF_') and options['ENABLED']:
                raise ValueError('OFF run unexpectedly enabled')
            if name.startswith('ZERO_') and (not options['ENABLED'] or options['FEATURE_WEIGHT'] != 0 or options['GEOMETRY_WEIGHT'] != 0):
                raise ValueError('nonzero coefficients in zero control')
            for key, path in [('checkpoint_sha256', CHECKPOINT), ('config_sha256', ROOT/'configs/VISION_test.yaml'), ('annotation_sha256', DATA/'annotations/train.json')]:
                if digest(path) != run[key]:
                    raise ValueError('changed locked runtime input: '+key)
                artifacts[str(path)] = digest(path)
            compared, unchanged, excluded = {}, [], []
            for file, historical in run['source_hashes'].items():
                if not file.startswith('gtr/') and file not in ENTRY_FILES:
                    continue
                observed = digest(ROOT/file)
                artifacts[str(ROOT/file)] = observed
                compared[file] = dict(recorded=historical, current=observed)
                if observed == historical:
                    unchanged.append(file)
                elif file == RETRIEVAL_ONLY:
                    excluded.append(file)
                else:
                    raise ValueError('default runtime source changed: '+file)
            if not ENTRY_FILES <= set(compared):
                raise ValueError('recorded default entry files unavailable')
            # Record a bounded direct-import check; do not claim a complete
            # Python dynamic-import proof or general retrieval-mode equality.
            imported = []
            for file in unchanged:
                for node in ast.walk(ast.parse((ROOT/file).read_text())):
                    if isinstance(node, ast.ImportFrom) and ((node.module or '').endswith('isolated_stream') or any(alias.name=='isolated_stream' for alias in node.names)):
                        imported.append(file)
                    if isinstance(node, ast.Import) and any(alias.name.endswith('isolated_stream') for alias in node.names):
                        imported.append(file)
            if imported:
                raise ValueError('changed retrieval module directly imported in compared default files')
            rows.append(dict(control='OFF' if name.startswith('OFF_') else 'ZERO', video_id=row['video_id'],
                original_source_commit=run['commit'], original_causal_online=run['causal_online'], original_full_video=run['full_video'],
                original_exact_predictions_retained=True, same_recorded_default_source_files=len(unchanged),
                compared_source_hashes=compared, changed_retrieval_only_files=excluded,
                direct_isolated_stream_imports_in_unchanged_files=imported))
    save_json(ROOT/'results/current_default_source_consistency_v1.json', dict(status='RECORDED_DEFAULT_SOURCE_MATCHES', rows=rows,
        controls_verified=len(rows), matching_source_file_counts=sorted({r['same_recorded_default_source_files'] for r in rows}),
        changed_retrieval_only_files=sorted({p for r in rows for p in r['changed_retrieval_only_files']}),
        source_artifacts=artifacts, new_model_load_performed=False, new_video_inference_performed=False,
        scope='all ten recorded complete OFF/zero control scenes retain original exact predictions and manifests; current GMT/checkpoint/YAML/TRAIN annotation and recorded GTR/default driver source bytes are checked; changed isolated_stream is explicitly separate',
        limitations='source consistency is not a new empirical run, library-binary identity, exhaustive dynamic-import proof or arbitrary retrieval/valid-pose behavior; original controls retain their original offline/causal protocol and source commit; newly added optional modules are not certified by this comparison', provenance=provenance()))
    print(json.dumps(dict(controls=len(rows), unchanged_source_counts=sorted({r['same_recorded_default_source_files'] for r in rows}),
                         excluded_changed_files=sorted({p for r in rows for p in r['changed_retrieval_only_files']}))))


if __name__ == '__main__':
    main()

"""Read-only GT provenance audit. Never edits labels or chooses evaluation results."""
from pathlib import Path
from collections import Counter
import argparse, configparser, hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--include-downloaded-test', action='store_true')
args = parser.parse_args()
track_root = ROOT/'code/GMT/TrackEval/data/gt/mot_challenge'
cross_root = ROOT/'code/GMT/MOTChallengeEvalKit_cv_test/data/eval/vision/gt'
test_root = ROOT/'datasets/VisionTrack/test'
names = (track_root/'seqmaps/vision-train.txt').read_text().splitlines()[1:]
assert len(names) == len(set(names))
if args.include_downloaded_test:
    state = json.loads((ROOT/'manifests/archive_intake_state.json').read_text())
    if not state.get('archives', {}).get('test.zip', {}).get('crc_verified_on_extract'):
        raise RuntimeError('Downloaded test archive has not completed extraction/CRC verification')


def read(path, delimiter):
    a = np.loadtxt(path, delimiter=delimiter, ndmin=2)
    if a.shape[1] < 6 or not np.isfinite(a).all():
        raise RuntimeError('Malformed GT: '+str(path))
    return a[:, :6], {
        'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'rows': len(a), 'columns': a.shape[1],
        'duplicate_frame_identity_pairs': len(a)-len(set(map(tuple, a[:, :2]))),
        'last_frame': int(a[:, 0].max()),
    }


def compare(a, b):
    ca, cb = Counter(map(tuple, a)), Counter(map(tuple, b))
    return {'equal_in_order': np.array_equal(a, b), 'equal_as_multiset': ca == cb,
            'first_only_rows': sum((ca-cb).values()),
            'second_only_rows': sum((cb-ca).values())}


records = []
for name in sorted(names):
    scene, view = name.rsplit('_', 1)
    a, am = read(track_root/'vision-train'/name/'gt/gt.txt', ',')
    b, bm = read(cross_root/scene/'gt'/f'{view}.txt', None)
    cfg = configparser.ConfigParser()
    cfg.read(track_root/'vision-train'/name/'seqinfo.ini')
    row = {'sequence': name, 'trackeval': am, 'crossview': bm,
           'trackeval_vs_crossview': compare(a, b),
           'trackeval_declared_frames': int(cfg['Sequence']['seqlength'])}
    if args.include_downloaded_test:
        c, cm = read(test_root/name/'gt/gt.txt', ',')
        row.update(downloaded=cm, downloaded_vs_trackeval=compare(c, a),
                   downloaded_vs_crossview=compare(c, b))
    records.append(row)
report = {'scope': 'GT provenance only; no labels, predictions, thresholds or metric formulas modified',
          'downloaded_included': args.include_downloaded_test, 'records': records}
if args.include_downloaded_test:
    actual = {p.name for p in test_root.iterdir() if p.is_dir()}
    report['downloaded_only_sequences'] = sorted(actual-set(names))
    report['official_seqmap_only_sequences'] = sorted(set(names)-actual)
suffix = 'downloaded' if args.include_downloaded_test else 'repository'
output = ROOT/f'reports/evaluation_input_comparison_{suffix}.json'
output.write_text(json.dumps(report, indent=2))
print(str(output))
print('Sequences:', len(records), 'repository GT mismatches:',
      sum(not x['trackeval_vs_crossview']['equal_in_order'] for x in records))

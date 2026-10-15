import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from collect_native_suites import main, validate_report
from common import sha256


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prediction = self.root/'predictions.json'
        self.prediction.write_text('[]')
        self.run = self.root/'manifest.json'
        self.run.write_text(json.dumps(dict(full_video=True, causal_online=True, video_id=1,
                                           predictions_sha256=sha256(self.prediction))))
        self.source = self.root/'sources.json'
        self.source.write_text(json.dumps(dict(video_predictions={'1': str(self.prediction)})))
        self.report = dict(status='PASS', full_video=True, causal_online=True, video_ids=[1],
            source_manifest_sha256=sha256(self.source), prediction_sha256={str(self.prediction): sha256(self.prediction)},
            run_manifests={'1': dict(path=str(self.run), sha256=sha256(self.run))}, raw_gt_sha256={})

    def test_import_requires_exact_evaluated_artifacts(self):
        validate_report(self.report, self.source, [1])
        self.prediction.write_text('[{}]')
        with self.assertRaisesRegex(ValueError, 'prediction files'):
            validate_report(self.report, self.source, [1])

    def test_partial_or_changed_manifest_is_rejected(self):
        partial = copy.deepcopy(self.report)
        partial['full_video'] = False
        with self.assertRaisesRegex(ValueError, 'complete declared'):
            validate_report(partial, self.source, [1])
        self.run.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'captured inference manifest'):
            validate_report(self.report, self.source, [1])

    def test_does_not_race_queue_owned_native_combination(self):
        manifests = self.root/'manifests'
        manifests.mkdir()
        (self.root/'results').mkdir()
        runtime = self.root/'runtime'
        runtime.mkdir()
        tag = 'causal_B5_dense_seed20261015_strict_suite_v1'
        (manifests/'identity_relations_matrix_v2.json').write_text(json.dumps(dict(methods=[dict(suite=tag, method_sources='unneeded.json')])) )
        (manifests/'geo_split_v1.json').write_text(json.dumps(dict(strict_development_video_ids=[1])))
        (manifests/'primary_vggt_inference_seed_pairs_v1.json').write_text(json.dumps(dict(combine_after_all_scenes='root-owned', inference_seeds=[20261015, 20261016])))
        with patch('collect_native_suites.ROOT', self.root), patch('collect_native_suites.RUNTIME', runtime), \
             patch('sys.argv', ['collect_native_suites.py', '--generate-ready']), \
             patch('collect_native_suites.ready', return_value=True) as ready, \
             patch('collect_native_suites.subprocess.run') as execute:
            main()
            ready.assert_not_called()
            execute.assert_not_called()
        self.assertFalse((self.root/'results'/f'{tag}_metrics.json').exists())


if __name__ == '__main__':
    unittest.main()

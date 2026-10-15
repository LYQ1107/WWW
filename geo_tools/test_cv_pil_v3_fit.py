"""Reject wrong/overlapping V3 TRAIN sources before opening any tensor."""
import json
from pathlib import Path
import tempfile
import unittest

from cv_pil_v3_fit import FIT_SCENES, load_fit
from cv_pil_native_handoff import sha256
from train_cv_pil_v3_matched import execute


class TestV3TrainAdmission(unittest.TestCase):
    def test_wrong_cohort_sha_rejected_before_source_access(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cohort.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'cohort SHA'):
                load_fit(p,'0'*64,d,d)

    def test_dev12_cannot_be_relabelled_as_train(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cohort.json'
            c=dict(cohort_version=3,native_feature_dim=1152,official_test_used=False,
                   used_teacher_results_for_selection=False,fit_scene_ids=sorted(FIT_SCENES),validation_scene_ids=[12],
                   fit=[dict(scene=12,key=[12,i,0]) for i in range(3000)])
            p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'frozen TRAIN scope'):
                load_fit(p,sha256(p),d,d)

    def test_teacher_selected_or_test_cohort_rejected_before_tensors(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cohort.json'
            for field in ('official_test_used','used_teacher_results_for_selection'):
                c=dict(cohort_version=3,native_feature_dim=1152,official_test_used=False,
                       used_teacher_results_for_selection=False,fit_scene_ids=sorted(FIT_SCENES),validation_scene_ids=[12])
                c[field]=True;p.write_text(json.dumps(c))
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Unauthorized TRAIN'):
                    load_fit(p,sha256(p),d,d)

    def test_training_refuses_unsealed_development_even_with_registered_steps(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);admission=root/'dev.json';admission.write_text('{"status":"RUNNING"}')
            p=root/'protocol.json';p.write_text(json.dumps(dict(status='PREREGISTERED_BEFORE_V3_TRAINING',
                seed=20261015,steps=300,development_admission=dict(path=str(admission),sha256=sha256(admission)))))
            with self.assertRaisesRegex(ValueError,'Complete real DEV admission'):
                execute(p,sha256(p),root/'never-created')
            self.assertFalse((root/'never-created').exists())


if __name__=='__main__':unittest.main()

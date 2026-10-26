"""Offline scheduler checks: no models launched and no real data mutated."""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("v5_scheduler", Path(__file__).with_name("run_worldobject_v5_controller.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SchedulingChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.runtime, self.root = self.base / "runtime", self.base / "outputs"
        self.runtime.mkdir()
        self.root.mkdir()
        module.write_json(self.runtime / "selection_launches.json", {
            "source_sha256": module.FROZEN_SOURCE, "jobs": {}})
        args = SimpleNamespace(repo=self.base, runtime=self.runtime, output_root=self.root,
            dry_run=True, python=Path("/usr/bin/python3"), mirror_root=self.base / "mirror",
            request_file=self.base / "human.txt", evaluation_python=Path("/usr/bin/python3"))
        self.controller = module.Controller(args)

    def tearDown(self):
        self.temp.cleanup()

    def test_process_start_identity_is_stable(self):
        first = module.process_identity(os.getpid())
        self.assertEqual(first, module.process_identity(os.getpid()))
        self.assertIsNone(module.process_identity(999999999))

    def test_original_process_is_adopted_without_duplicate(self):
        identity = module.process_identity(os.getpid())
        module.write_json(self.runtime / "selection_launches.json", {
            "source_sha256": module.FROZEN_SOURCE, "jobs": {"B3": {
                "pid": os.getpid(), "command": identity["argv"], "device": "cpu", "GPU_UUID": None}}})
        self.controller.adopt_selections()
        with patch.object(module.subprocess, "Popen") as popen:
            self.assertFalse(self.controller.launch("train_B3_selection", ["never-run"]))
            popen.assert_not_called()
        self.assertTrue(self.controller.running("train_B3_selection"))

    def test_pid_reuse_does_not_adopt_different_command(self):
        module.write_json(self.runtime / "selection_launches.json", {
            "source_sha256": module.FROZEN_SOURCE, "jobs": {"B3": {
                "pid": os.getpid(), "command": ["different-process"]}}})
        with self.assertRaisesRegex(ValueError, "DIFFERENT_PROCESS"):
            self.controller.adopt_selections()

    def test_no_test_registration_without_all_final_gates(self):
        with patch.object(self.controller, "storage_ready", return_value=True), \
             patch.object(self.controller, "register_test") as register, \
             patch.object(self.controller, "training", return_value=False), \
             patch.object(module, "gpu_snapshot", return_value=[]), \
             patch("worldmot.runtime.world_object_v5_contract.v5_source_fingerprint", return_value=module.FROZEN_SOURCE):
            self.controller.tick()
            register.assert_not_called()

    def test_occupied_gpu_never_counts_as_idle(self):
        rows = [{"uuid": "used", "name": "Tesla V100", "free_MiB": 30000, "occupied": True},
                {"uuid": "free", "name": "Tesla V100", "free_MiB": 30000, "occupied": False}]
        with patch.object(module, "gpu_snapshot", return_value=rows):
            self.assertEqual([x["uuid"] for x in self.controller.free_gpus()], ["free"])

    def test_pending_cuda_initialization_reserves_device(self):
        self.controller.state["jobs"]["x"] = {"status": "RUNNING", "GPU_UUID": "pending"}
        rows = [{"uuid": "pending", "name": "Tesla V100", "free_MiB": 30000, "occupied": False}]
        with patch.object(module, "gpu_snapshot", return_value=rows):
            self.assertEqual(self.controller.free_gpus(), [])

    def test_failed_job_requires_review_before_retry(self):
        self.controller.state["jobs"]["x"] = {"status": "EXITED", "pid": 10, "exit_code": 1, "result": None}
        with patch.object(module.subprocess, "Popen") as popen:
            self.assertFalse(self.controller.launch("x", ["never-run"]))
            popen.assert_not_called()
        self.assertIn("x", self.controller.state["engineering_attention"])

    def test_unknown_exit_cannot_bypass_review_by_resume(self):
        self.controller.state["jobs"]["x"] = {"status": "EXITED", "pid": 10, "exit_code": None, "result": None}
        self.assertFalse(self.controller.launch("x", ["never-run", "--resume"], resume=True))
        self.assertIn("x", self.controller.state["engineering_attention"])

    def test_registered_normal_extension_can_resume(self):
        self.controller.state["jobs"]["x"] = {"status": "EXITED", "pid": 10, "exit_code": None, "result": None}
        with patch.object(self.controller, "storage_ready", return_value=True):
            self.assertTrue(self.controller.launch("x", ["never-run", "--resume"], resume=True, registered_continuation=True))
        self.assertEqual(len(self.controller.state["dry_run_launches"]), 1)

    def test_two_matched_geometry_finals_wait_for_gpu_while_B2_can_use_cpu(self):
        module.write_json(self.controller.private / "common_train_budget.json", {"epochs": 12, "v5_source_sha256": module.FROZEN_SOURCE})
        for v in module.TRAINED:
            module.write_json(self.root / f"train/{v}/selection/selection_gate.json",
                              {"epochs": 12, "v5_source_sha256": module.FROZEN_SOURCE})
        with patch.object(self.controller, "free_gpus", return_value=[]), patch.object(self.controller, "launch") as launch:
            self.assertFalse(self.controller.training())
            self.assertEqual(launch.call_count, 1)
            args, kwargs = launch.call_args
            self.assertEqual(args[0], "train_B2_final")
            self.assertIsNone(args[2])
            self.assertEqual(args[1][args[1].index("--device") + 1], "cpu")

    def test_partial_scope_is_preserved(self):
        self.controller.private.mkdir()
        module.write_json(self.controller.private / "scope_v5_test.json", {"partial": True})
        with self.assertRaisesRegex(ValueError, "PARTIAL_TEST_REGISTRATION"):
            self.controller.register_test()

    def test_native_scene_names_and_full22_coverage_are_validated(self):
        audit_path = self.controller.args.mirror_root / "fulltrain_20261022/audit/data_audit.json"
        module.write_json(audit_path, {"test": {"scenes": [
            {"scene": s, "name": "native_name_" + str(s)} for s in range(1, 23)]}})
        metric = {"status": "COMPLETE_VALID_SCOPE", "raw_GT_modified": False,
                  "evaluated_scenes": ["native_name_" + str(s) for s in module.LEGAL],
                  "coverage": [{"scene": s, "frames": 1 if s < 22 else 28998} for s in range(1, 23)],
                  "blocked": [{"scene": s, "reason": "RAW_GT_DUPLICATE_OR_VERSION_INVALID"}
                              for s in range(1, 23) if s not in module.LEGAL]}
        path = self.root / "evaluation/B0/metrics.json"
        module.write_json(path, metric)
        self.assertEqual(self.controller.evaluation({v: [] for v in module.VARIANTS}), ["B0"])
        metric["coverage"].pop()
        module.write_json(path, metric)
        with self.assertRaisesRegex(ValueError, "LEGAL_ORIGINAL_GT14_SCOPE_REQUIRED"):
            self.controller.evaluation({v: [] for v in module.VARIANTS})


if __name__ == "__main__":
    unittest.main()

"""Generated-only checks for resource attribution, PID reuse and missing evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import monitor_worldobject_v5_gpu_occupancy as observer


def process(root, pid, parent, start):
    folder = root / str(pid)
    folder.mkdir(exist_ok=True)
    fields = ["S", str(parent)] + ["0"] * 17 + [str(start)]
    (folder / "stat").write_text(str(pid) + " (worker with spaces) " + " ".join(fields))


class Checks(unittest.TestCase):
    def test_ancestry_and_reused_leader_PID(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            process(root, 3453, 1, 10)
            process(root, 4001, 3453, 20)
            self.assertTrue(observer.belongs_to_goal(4001, {3453: "10"}, root))
            process(root, 3453, 1, 99)
            self.assertFalse(observer.belongs_to_goal(4001, {3453: "10"}, root))
            self.assertFalse(observer.belongs_to_goal(9001, {3453: "10"}, root))

    def test_shared_and_idle_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            process(root, 3453, 1, 10)
            process(root, 40000001, 3453, 20)
            process(root, 50000001, 1, 30)
            devices = observer.occupancy("0, GPU-a, Tesla V100\n1, GPU-b, Tesla V100\n",
                                         "40000001, GPU-a, 1000\n50000001, GPU-a, 9000\n", {3453: "10"}, root)
            self.assertTrue(devices[0]["goal_contention_observed"])
            self.assertFalse(devices[0]["compute_idle_at_sample"])
            self.assertEqual(devices[0]["other_process_count"], 1)
            self.assertTrue(devices[1]["compute_idle_at_sample"])
            self.assertNotIn("50000001", json.dumps(devices))

    def test_invalid_queries_cannot_report_idle(self):
        def timeout(_):
            raise subprocess.TimeoutExpired("nvidia-smi", 15)
        observed = observer.sample({3453: "10"}, runner=timeout)
        self.assertFalse(observed["valid"])
        self.assertIsNone(observed["goal_contention_observed"])
        self.assertNotIn("devices", observed)
        with self.assertRaises(ValueError):
            observer.occupancy("0, GPU-a, Tesla V100\n", "4001, GPU-a, N/A\n", {3453: "10"})
        with self.assertRaises(ValueError):
            observer.occupancy("0, GPU-a, Tesla V100\n", "4001, GPU-unknown, 100\n", {3453: "10"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt = {"status": "PASSED_GENERATED_ONLY_GPU_OBSERVER_CHECKS" if result.wasSuccessful() else "FAILED",
               "executed_UTC": datetime.now(timezone.utc).isoformat(),
               "checks": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "observer_source_sha256": hashlib.sha256(Path(observer.__file__).read_bytes()).hexdigest(),
               "check_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "GT_read": False, "model_forward_calls": 0, "real_GPU_queries_in_checks": False}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)

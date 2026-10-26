"""Synthetic corruption and recovery checks; never touch real models or NAS."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from backup_worldobject_v5_durable_work import (
    FROZEN_SOURCE, checkpoint_files, make_archive, publish_snapshot, sha_bytes)


class DurableBackupChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.work = Path(self.temporary.name) / "work"
        self.base = self.work / "train/B3/selection"
        self.base.mkdir(parents=True)
        self.contents = b"SYNTHETIC_CHECKPOINT_WITH_OPTIMIZER_AND_RNG"
        self.pointer = {"checkpoint": "resume_00001.pt", "sha256": sha_bytes(self.contents),
                        "v5_source_sha256": FROZEN_SOURCE, "scope_sha256": "synthetic-scope",
                        "resume_boundary": "COMPLETE_SCENE_ONLY"}
        (self.base / "resume_00001.pt").write_bytes(self.contents)
        (self.base / "RESUME.json").write_text(json.dumps(self.pointer))
        (self.work / "provenance").mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_corrupt_checkpoint_cannot_be_archived(self):
        (self.base / "resume_00001.pt").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "BACKUP_SHA_MISMATCH"):
            checkpoint_files(self.work, "B3", "selection")

    def test_archive_preserves_checkpoint_and_resume_pointer_bytes(self):
        info, files, manifest = checkpoint_files(self.work, "B3", "selection")
        with tarfile.open(fileobj=io.BytesIO(make_archive(files)), mode="r:gz") as archive:
            restored = archive.extractfile("train/B3/selection/" + info["checkpoint"]).read()
            self.assertEqual(restored, self.contents)
            self.assertEqual(json.loads(archive.extractfile("BACKUP_MANIFEST.json").read()), manifest)
            self.assertEqual(sha_bytes(restored), info["sha256"])

    def test_final_gate_and_actual_final_weights_are_backed_up(self):
        final = self.base.with_name("final")
        final.mkdir()
        for name in ("resume_00001.pt", "RESUME.json"):
            (final / name).write_bytes((self.base / name).read_bytes())
        weights = final.parent / "membership_v5.pt"
        weights.write_bytes(b"SYNTHETIC_FINAL_WEIGHTS")
        gate = {"checkpoint": weights.name, "checkpoint_sha256": sha_bytes(weights.read_bytes())}
        (final.parent / "train_gate.json").write_text(json.dumps(gate))
        _, files, _ = checkpoint_files(self.work, "B3", "final")
        self.assertEqual(files["train/B3/membership_v5.pt"], weights.read_bytes())
        self.assertIn("train/B3/train_gate.json", files)

    def test_remote_readback_required_and_existing_different_bytes_preserved(self):
        destination = Path(self.temporary.name) / "synthetic-remote"
        receipt = publish_snapshot(destination, "snapshot.tar.gz", self.contents)
        self.assertEqual(receipt["sha256"], sha_bytes(self.contents))
        with self.assertRaisesRegex(ValueError, "READBACK_SHA_MISMATCH"):
            publish_snapshot(destination, "snapshot.tar.gz", b"different-data")
        self.assertEqual((destination / "snapshot.tar.gz").read_bytes(), self.contents)


if __name__ == "__main__":
    unittest.main()

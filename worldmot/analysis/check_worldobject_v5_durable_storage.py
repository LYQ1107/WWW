"""Real local filesystem checks for disconnect-safe archiving; no NAS or GT."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from archive_worldobject_v5_predictions import archive_scene, FROZEN_SOURCE
from summarize_worldobject_v5 import cached_runtime
from worldobject_v5_durable_storage import (
    pinned_nas_root, fd_mount_id, verified_copy, digest, replace_local_with_verified_symlink,
    resumable_verified_copy, write_json)


class DurableStorageChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.mount = self.base / "synthetic-mount"
        self.root = self.mount / "experiment"
        self.root.mkdir(parents=True)
        fd = os.open(self.mount, os.O_RDONLY | os.O_DIRECTORY)
        try:
            mount_id = fd_mount_id(fd)
        finally:
            os.close(fd)
        self.identity = {"mount_id": mount_id, "mountpoint": str(self.mount),
                         "filesystem": "fuse.sshfs", "source": "synthetic"}

    def tearDown(self):
        self.temp.cleanup()

    def test_absent_nas_mount_cannot_write_underlying_local_directory(self):
        with patch("worldobject_v5_durable_storage.mount_record", return_value=None):
            with self.assertRaisesRegex(OSError, "VERIFIED_NAS_MOUNT_REQUIRED"):
                with pinned_nas_root(self.root, self.mount):
                    self.fail("A missing mount cannot yield a writable path")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_wrong_mount_descriptor_rejected_before_writes(self):
        with patch("worldobject_v5_durable_storage.mount_record", return_value={**self.identity, "mount_id": -1}):
            with self.assertRaisesRegex(OSError, "BEFORE_DIRECTORY_ANCHOR"):
                with pinned_nas_root(self.root, self.mount):
                    self.fail("Wrong filesystem cannot be accepted")

    def test_held_directory_cannot_fall_through_after_path_replacement(self):
        moved = self.mount / "original-experiment"
        with patch("worldobject_v5_durable_storage.mount_record", return_value=self.identity):
            with pinned_nas_root(self.root, self.mount) as (pinned, _):
                self.root.rename(moved)
                self.root.mkdir()
                (pinned / "proof.bin").write_bytes(b"WRITTEN_ON_ORIGINAL_DIRECTORY")
        self.assertFalse((self.root / "proof.bin").exists())
        self.assertEqual((moved / "proof.bin").read_bytes(), b"WRITTEN_ON_ORIGINAL_DIRECTORY")

    def test_mount_changed_during_copy_cannot_report_success(self):
        with patch("worldobject_v5_durable_storage.mount_record", side_effect=[self.identity, None]):
            with self.assertRaisesRegex(OSError, "DURING_COPY_READBACK"):
                with pinned_nas_root(self.root, self.mount) as (pinned, _):
                    (pinned / "pending.bin").write_bytes(b"pending")

    def test_copy_preserves_collision_and_retries_own_incomplete_temporary(self):
        source = self.base / "source.bin"
        target = self.root / "target.bin"
        source.write_bytes(b"complete")
        temporary = target.with_name(target.name + ".partial." + str(os.getpid()))
        temporary.write_bytes(b"incomplete")
        verified_copy(source, target, digest(source))
        self.assertEqual(target.read_bytes(), b"complete")
        self.assertEqual(len(list(self.root.glob("*.failed.*"))), 1)
        source.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "EXISTING_NAS_OUTPUT_DIFFERS"):
            verified_copy(source, target, digest(source))
        self.assertEqual(target.read_bytes(), b"complete")

    def test_atomic_directory_exchange_preserves_readers_and_bytes(self):
        folder = self.base / "local-scene"
        remote = self.root / "remote-scene"
        folder.mkdir()
        remote.mkdir()
        for directory in (folder, remote):
            (directory / "tracks.jsonl").write_bytes(b"IDENTICAL_PREDICTION\n")
        with (folder / "tracks.jsonl").open("rb") as reader:
            replace_local_with_verified_symlink(folder, remote)
            self.assertTrue(folder.is_symlink())
            self.assertEqual(reader.read(), (folder / "tracks.jsonl").read_bytes())
        self.assertFalse(list(self.base.glob("local-scene.verified-migration.*")))

    def test_complete_scene_archives_all_original_output_bytes_without_modification(self):
        folder = self.base / "local-scene"
        folder.mkdir()
        names = ("tracks.jsonl", "timings.jsonl", "birth_audit.jsonl")
        for name in names:
            (folder / name).write_text("SYNTHETIC " + name + "\n")
        manifest = {"status": "COMPLETE", "variant": "B1", "scene": 1,
                    "v5_source_sha256": FROZEN_SOURCE, "GT_in_inference": False, "causal": True,
                    "output_sha256": {name: digest(folder / name) for name in names}}
        (folder / "manifest.json").write_text(json.dumps(manifest))

        @contextmanager
        def local_synthetic_anchor(root):
            yield root, self.identity

        with patch("archive_worldobject_v5_predictions.pinned_nas_root", local_synthetic_anchor):
            receipt = archive_scene(folder, self.root, "B1", 1)
        remote = self.root / "predictions/B1/scene_01"
        self.assertEqual(receipt["status"], "COMPLETE_NAS_SCENE_OUTPUT_READBACK_VERIFIED")
        for name in (*names, "manifest.json"):
            self.assertEqual((remote / name).read_bytes(), (folder / name).read_bytes())
        self.assertTrue(folder.is_dir())
        self.assertFalse(receipt["local_files_reclaimed"])

    def test_mixed_storage_timings_are_explicitly_not_a_controlled_speed_ranking(self):
        provenance = self.base / "provenance"
        provenance.mkdir()
        (provenance / "prediction_output_storage_modes.json").write_text(json.dumps({
            "per_variant_scene": {"B0": {"1": "NAS_DIRECT", "2": "LOCAL_DURABLE"}}}))
        for scene in (1, 2):
            folder = self.base / f"predictions/B0/scene_{scene:02d}"
            folder.mkdir(parents=True)
            (folder / "manifest.json").write_text(json.dumps({"status": "COMPLETE", "worker_wall_s": 2.,
                "device": "synthetic", "peak_RSS_bytes": 100, "CUDA_peak_allocated_bytes": 0}))
            (folder / "timings.jsonl").write_text(json.dumps({"complete_timestamp_s": 1.}) + "\n")
        result = cached_runtime(self.base, "B0")
        self.assertEqual(result["original_output_storage_scene_counts"], {"NAS_DIRECT": 1, "LOCAL_DURABLE": 1})
        self.assertFalse(result["uniform_recorded_local_output_storage"])
        self.assertTrue(result["cached_FPS_not_a_controlled_algorithm_speed_ranking"])
        self.assertEqual(result["CACHED_single_worker_step_FPS"], 1.)
        self.assertIsNone(result["LIVE_RGB_to_ID_FPS"])

    def interrupted_chunk_copy(self):
        source, target, state = self.base / "chunk-source", self.root / "chunk-target", self.base / "copy-state.json"
        source.write_bytes(b"abcdefghijklmnopq")
        original_write = write_json

        def interrupt_after_first_ack(path, value):
            original_write(path, value)
            raise OSError("SYNTHETIC_DISCONNECT_AFTER_FIRST_VERIFIED_CHUNK")

        with patch("worldobject_v5_durable_storage.write_json", side_effect=interrupt_after_first_ack):
            with self.assertRaisesRegex(OSError, "SYNTHETIC_DISCONNECT"):
                resumable_verified_copy(source, target, digest(source), state, chunk_bytes=4)
        self.assertEqual(json.loads(state.read_text())["committed_bytes"], 4)
        self.assertFalse(target.exists())
        return source, target, state

    def test_large_copy_resumes_after_ack_without_retransmitting_committed_ranges(self):
        source, target, state = self.interrupted_chunk_copy()
        proof = resumable_verified_copy(source, target, digest(source), state, chunk_bytes=4)
        self.assertEqual(target.read_bytes(), source.read_bytes())
        chunks = json.loads(state.read_text())["chunks"]
        self.assertEqual([chunk["offset"] for chunk in chunks], [0, 4, 8, 12, 16])
        self.assertEqual(proof["verified_bytes"], 17)
        self.assertEqual(proof["chunk_count"], 5)
        self.assertTrue(proof["final_native_or_adoption_full_SHA_still_required"])

    def test_changed_remote_prefix_never_becomes_a_completed_archive(self):
        source, target, state = self.interrupted_chunk_copy()
        partial = next(self.root.glob("chunk-target.verified-upload.*"))
        partial.write_bytes(b"xxxx")
        with self.assertRaisesRegex(ValueError, "REMOTE_COMMITTED_PREFIX_CHANGED"):
            resumable_verified_copy(source, target, digest(source), state, chunk_bytes=4)
        self.assertFalse(target.exists())
        self.assertEqual(source.read_bytes(), b"abcdefghijklmnopq")

    def test_unacknowledged_tail_is_retried_and_full_byte_equality_is_preserved(self):
        source, target, state = self.interrupted_chunk_copy()
        partial = next(self.root.glob("chunk-target.verified-upload.*"))
        with partial.open("ab") as stream:
            stream.write(b"UNACKNOWLEDGED_NETWORK_TAIL")
        resumable_verified_copy(source, target, digest(source), state, chunk_bytes=4)
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertTrue(json.loads(state.read_text())["complete"])

    def test_unrelated_existing_file_cannot_reuse_a_partial_copy_proof(self):
        source, target, state = self.interrupted_chunk_copy()
        target.write_bytes(b"different-existing")
        with self.assertRaisesRegex(ValueError, "EXISTING_NAS_OUTPUT_DIFFERS"):
            resumable_verified_copy(source, target, digest(source), state, chunk_bytes=4)
        self.assertEqual(target.read_bytes(), b"different-existing")

    def test_remote_copy_proof_stays_immutable_after_later_reverification(self):
        source, target, state = self.interrupted_chunk_copy()
        first = resumable_verified_copy(source, target, digest(source), state, chunk_bytes=4)
        original = Path(first["immutable_contiguous_remote_range_proof"])
        proof_sha = first["contiguous_remote_range_proof_sha256"]
        second = resumable_verified_copy(source, target, digest(source), state, chunk_bytes=4)
        self.assertEqual(digest(original), proof_sha)
        self.assertTrue(Path(second["immutable_contiguous_remote_range_proof"]).is_file())
        self.assertEqual(target.read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main()

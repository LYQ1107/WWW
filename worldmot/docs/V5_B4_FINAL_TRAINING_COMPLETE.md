WorldMOT V5: B4 complete fresh all 24 final training

Verified UTC: 2026-10-27T10:04:17.273170+00:00. B4 completed the registered twelve final epochs from the common fresh initialization. It is the new control with a dedicated head and all explicit Ω geometry disabled. It has not been run or scored on TEST. B2 and B3 continue their original healthy training processes; canonical new-model TEST still requires all three actual final gates.

Every epoch covered all 24 TRAIN scenes, 28,754 paired times and 557,002 frozen candidates. Total exposures were 345,048 paired times and 6,684,024 candidates. The independent verification read all 288 actual scene logs and all 48 frozen input manifests, checked each scene's timestamps/candidates against the original TRAIN scope, and used the strict frozen checkpoint loader on CPU. All gradients and FP32 parameters were finite. Each scene reset once at its origin; there were no internal ID resets or GT repairs.

| Artifact | SHA-256 |
|---|---|
| Final membership_v5.pt | 4c0bb1f46b91cc759ce46d23f04e9b30e7b2a93369cefdead7c4648de6e3e0ab |
| Full24 final train gate | d9a95284c59252d438fff34a10a2cd060e1556a8b62b80eba04ee66b2b32cda8 |
| NAS final training archive | d737eee6612bce0ef69378615f5451751e6d486ed4cad1f1ace48931c905e533 |

The dedicated membership/NULL network has 606,468 parameters, seed 20261027 and 86,376 actual AdamW updates. Its initial tensor digest remained the registered 3fb3f3d6…; the final tensor digest is fc8c9c75bd71b87be52e2bef5d80399eda2bf577da2b1713a0b12209e209ce97. The final resume payload was independently loaded and its model tensors exactly matched the delivered head. Its completed 12-epoch history matched the gate, and optimizer state plus Python/NumPy/Torch/CUDA RNG states were present. Verification created no optimizer and ran no forward pass.

The progress file's epoch 13 is the index after the loop: completed_epochs=12 and complete=true. It does not indicate a thirteenth training epoch. The final TRAIN membership loss was 0.05984688. TRAIN epoch compute wall time summed to 15947.1 seconds; this is neither inference FPS nor evidence of TEST gain or generalization.

The actual NAS fuse.sshfs mount 100 received a 10,790,245-byte bundle. All 305 internal files were checked against their byte hashes, then the entire remote archive was read back and matched SHA-256. It includes the final head/gate, completed resume payload, all 288 logs, frozen provenance, strict verification receipt and the exact b9c2d5b source snapshot.

NAS archive: `/data1/liuyeqiang/nas/A100/WorldMOT/worldobject_v5_decoupled_birth_20261027/final_training/B4_final_full24_4c0bb1f46b91cc75_d737eee6612bce0e.tar.gz`.

[Machine-readable completion and storage proof](../results/v5_B4_full24_final_training_completion.json) retains complete hashes and per-file coverage. [The common-budget record](V5_COMMON12_AND_FINAL_TRAINING.md) explains why the frozen budget remained 12. No result from historical No-Geometry V3 is assigned to this new B4 checkpoint. Formal 22-video inference, original legal 14 TrackEval/MATLAB, paired attribution and real RGB timing remain pending for B2/B3/B4.

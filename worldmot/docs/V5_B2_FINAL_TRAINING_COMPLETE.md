# WorldMOT V5 B2 final TRAIN completion

B2 completed the preregistered fresh 12-epoch full-TRAIN run. B2 uses the dedicated membership head with the old 16-dimensional geometry padded to 32 dimensions; it is not the B3 rich-geometry model or GMT. Each epoch covered all 24 TRAIN scenes, 28,754 paired timestamps and 557,002 candidates: 345,048 timestamp exposures and 6,684,024 candidate exposures in total.

The strict verification checked all 288 actual scene logs and 48 input manifests, the original TRAIN annotation SHA, the registered seeded scene order, finite FP32 parameters and gradients, no internal scene identity resets and no GT identity repairs. The model has 606,468 parameters, seed 20261027 and 86,376 optimizer updates. Verification used a CPU frozen loader, with no model forward, optimizer creation or TEST data access.

- Final head SHA-256: `91e5c2792f215745d5ece7a918458371ee1f289107bbc3f326a42cb75c96e827`
- Final gate SHA-256: `02d4d97b9ac5f1d43ea03fdb10e760d5b04d2ce48980a9c25ee5cd507d3682ac`
- Final resume payload SHA-256: `fa59f8eb95c5517ebfdec5cfe762a96dd1001f1e94afb5971f7e7c4bf4ab42b0`
- Trained tensor SHA-256: `80f10acb9c2ea8c26f9d3ed0786983407ae4c97836ba93bd1522286b4140062f`; initial tensor SHA-256: `3fb3f3d6336e8aa32d635ecff3c05929b0d4b8f265f4a0ccf56771814d13ed23`.
- NAS archive SHA-256: `79f1dbae77a2c2be7474d4f191a5dd1e36d76ae914651e3fd084187278bdc077`. All 310 members were checked and the full remote archive was read back from the verified NAS mount.

The epoch-12 TRAIN membership loss is 0.06250032686228624, and the summed TRAIN scene compute wall time is 24,945.52 seconds. Neither is an inference FPS, a TEST score or proof of generalization. At archival, B3 was still training and the registered all-three-final-gates barrier had not opened; no B2/B3/B4 formal TEST improvement is claimed.

See the [completion and archive receipt](../results/v5_B2_full24_final_training_completion.json) for the complete provenance and verification details.

# V5 B3 full TRAIN completion

B3 completed the preregistered fresh 12-epoch training on all 24 TRAIN scenes. The independent verifier checked 288 scene logs and 48 input manifests, with 28,754 paired timestamps and 557,002 candidates in each epoch. Gradients and final parameters were finite, and no scene-internal identity reset or ground-truth identity repair was found. The verification did not read TEST GT or run model inference.

The frozen checkpoint SHA-256 is `5c29624a09c2dec697adea55688a34882698e5354278f7ed83297d04e36a0e9a`; its gate SHA-256 is `75b8d5fbffc4613ff1765f929873e491de33ba99745e321fbbb09b01e4844f95`. The matched initial tensor SHA-256 is `3fb3f3d6336e8aa32d635ecff3c05929b0d4b8f265f4a0ccf56771814d13ed23`. The final TRAIN membership loss was 0.0684923220; this is not a TEST accuracy measure.

The complete checkpoint, resume state, source snapshot, verification evidence and training logs were archived on the verified NAS mount. Full readback SHA-256: `03772d1da033090dd82cacf23f90df0843e50096d0770e38d2ca9012e866f173`. The archive contains 310 individually verified files.

After all three final TRAIN gates existed, the registered controller entered full 22-scene causal TEST and began B3 inference. Formal B2/B3/B4 TEST results are pending. The legal 14-scene native evaluations must complete before drawing conclusions about GMT or geometry contribution.

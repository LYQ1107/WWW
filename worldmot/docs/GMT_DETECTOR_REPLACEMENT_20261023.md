# GMT pre-GTA detector replacement

Status: IMPLEMENTING. Base: `9a5572ebb3184b58253d448648195d175983ca1c`.

The frozen GMT configuration uses RGB, DLA/BiFPN and CenterNet proposals,
`ROI_HEADS.NO_BOX_HEAD=True`, and an association candidate score threshold
of 0.525. The earliest native detection set is therefore the current-image
CenterNet proposals filtered at this threshold, before track ID creation.
Extracting these candidates does not require calling the GTA transformer.

The native `_forward_asso()` also jitters ROI boxes for its appearance
descriptor. This random appearance operation must not change the detector
candidate set. The independent adapter extracts its explicitly labeled frozen
appearance feature from the unjittered current boxes for reproducible A1/V2
features. Native detection parity will be tested on real images.

Required audits: official RGB preprocessing parity, invertible input/box map,
person-only/current images, both views, single image versus prefix equality,
zero GTA calls, checkpoint/config/source SHA, original candidate counts and
ROI dimensions. Test selection is forbidden. Shared immutable caches store
scene/frame/camera/image IDs and stable detection indices without GT identity.

Milestones: B-M0 adapter/cache; B-M1 TRAIN detector comparison; B-M2 V1;
B-M3 full 24-scene object cache and association training; B-M4 frozen TEST;
B-M5 official comparison/error attribution. All results remain NOT_AVAILABLE
until backed by actual completed artifacts.

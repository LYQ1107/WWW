# Reprojection

The projection module takes stored world support points and the current camera model, applies the camera-from-world transform, rejects negative depth, maps model pixels back through the recorded crop/resize homography, clips to the original image, and emits a projected 2-D box and visibility fraction. Runtime output keeps `native_view_mask`, `projected_from_world`, and `projection_refined_by_target_view` as separate fields.

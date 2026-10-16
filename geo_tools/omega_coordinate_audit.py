"""Real first24 images and original native detector boxes, without model/GT."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from omega_common import ROOT, RUNTIME, NATIVE, jsonl, provenance, save_new, sha256
from omega_modules import geometry_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = ROOT/'manifests/omega_fixed_pair_cohort_v1.json'
    cohort = json.loads(manifest_path.read_text())
    pairs = [p for p in cohort['pairs'] if p['stage'] == 'initial24']
    adapter = geometry_module('omega_adapter')
    image_ids = {i['image_id'] for p in pairs for i in p['images']}
    boxes = collections.defaultdict(list)
    for scene in sorted({p['scene'] for p in pairs}):
        for row in jsonl(NATIVE/f'scene{scene}_full_v1/detector_provenance.jsonl.gz'):
            image_id = row['source'][0]
            if image_id in image_ids:
                x, y, w, h = row['detection']['bbox']
                boxes[image_id].append([x, y, x+w, y+h])
    rows = []
    count, largest = 0, 0.
    for pair in pairs:
        for image in pair['images']:
            assert sha256(image['path']) == image['sha256']
        for resolution in (416, 512):
            tensor, mappings = adapter.preprocess_pair(RUNTIME/'official_vggt_omega', [i['path'] for i in pair['images']], resolution)
            images = []
            for image, mapping in zip(pair['images'], mappings):
                corners = []
                for x1, y1, x2, y2 in boxes[image['image_id']]:
                    # Continuous detector locations are diagnostic probes only;
                    # crop validity masks are retained, never clipped into validity.
                    corners.extend([[x1, y1], [x2, y2], [(x1+x2)/2, (y1+y2)/2]])
                points = np.array(corners, dtype=float).reshape(-1, 2)
                mapped, valid = mapping.forward(points)
                back, inverse_valid = mapping.inverse(mapped)
                np.testing.assert_array_equal(valid, inverse_valid)
                error = float(np.max(np.abs(back[valid]-points[valid]))) if valid.any() else None
                if error is not None:
                    assert error < 1e-9
                    largest = max(largest, error)
                count += len(points)
                images.append(dict(image_id=image['image_id'], camera_id=image['camera_id'],
                                   original_sha256=image['sha256'], mapping=mapping.record(),
                                   native_detection_boxes=len(boxes[image['image_id']]),
                                   tested_corner_center_points=len(points), valid_points=int(valid.sum()),
                                   invalid_points=int((~valid).sum()), max_inverse_error=error))
            rows.append(dict(pair_id=pair['pair_id'], resolution=resolution, actual_tensor_shape=list(tensor.shape), images=images))
    save_new(args.output, dict(status='PASS_REAL_IMAGE_PREPROCESSING_AND_BOX_COORDINATES_ONLY',
             cohort=dict(path=str(manifest_path), sha256=sha256(manifest_path)), pairs=24,
             resolutions=[416, 512], point_instances=count, largest_inverse_error=largest, rows=rows,
             source_geometry_module_sha256=sha256(ROOT/'gtr/geometry/omega_adapter.py'),
             no_weights_loaded=True, no_GT_opened=True, real_Omega_forward='NOT_EXECUTED',
             reliability='NOT_MEASURED', provenance=provenance()))
    print(json.dumps(dict(status='PASS_REAL_IMAGE_COORDINATES', point_instances=count, largest_inverse_error=largest)))


if __name__ == '__main__':
    main()

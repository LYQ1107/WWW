"""Critical coordinate/pose tests; no fake model success or GMT inference."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np
from PIL import Image
from omega_common import ROOT, RUNTIME

# Load the independent geometry package without importing the unrelated GMT
# registration side effects in gtr/__init__.py or requiring a GPU.
package = types.ModuleType('omega_geometry_test_package')
package.__path__ = [str(ROOT / 'gtr/geometry')]
sys.modules[package.__name__] = package
def module(name):
    full = package.__name__ + '.' + name
    spec = importlib.util.spec_from_file_location(full, ROOT / 'gtr/geometry' / (name+'.py'))
    result = importlib.util.module_from_spec(spec)
    sys.modules[full] = result
    spec.loader.exec_module(result)
    return result

adapter = module('omega_adapter')
geometry = module('omega_geometry')
alignment = sys.modules[package.__name__ + '.geometry_alignment']
SOURCE = RUNTIME / 'official_vggt_omega'


class Tests(unittest.TestCase):
    def test_original_projection_crop_padding_and_camera_swap(self):
        rng = np.random.default_rng(20261016)
        X = rng.normal(size=(100, 3)); X[:, 2] += 8
        theta = .14
        R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        E = np.array([np.c_[np.eye(3), [0, 0, 0]], np.c_[R, [-.8, .1, .05]]])
        K = np.array([[[500, 0, 320], [0, 490, 224], [0, 0, 1]],
                      [[530, 0, 320], [0, 510, 224], [0, 0, 1]]], float)
        maps = [adapter.PixelMapping(1600, 900, 40, 20, 1500, 850, 624, 352, 8, 48, 640, 448),
                adapter.PixelMapping(900, 1600, 10, 50, 880, 1500, 256, 432, 192, 8, 640, 448)]
        points = []
        for e, k, m in zip(E, K, maps):
            homogeneous = (X @ e[:, :3].T + e[:, 3]) @ k.T
            network = homogeneous[:, :2]/homogeneous[:, 2:]
            original = alignment.project_points(np.linalg.inv(m.A), network)
            points.append(original)
        result = geometry.original_fundamental(E, K, maps)
        self.assertLess(np.max(alignment.symmetric_epipolar_distance(result['F_original'], *points)), 1e-9)
        reverse = geometry.original_fundamental(E[::-1], K[::-1], maps[::-1])
        self.assertLess(geometry.fundamental_equivalence(result['F_original'].T, reverse['F_original']), 1e-12)
        self.assertEqual(result['reliability'], 'NOT_MEASURED')

    def test_global_rigid_gauge_does_not_change_F(self):
        E = np.array([np.c_[np.eye(3), [0, 0, 0]], np.c_[np.eye(3), [1, .1, 0]]], float)
        K = np.array([np.diag([400, 400, 1])]*2, float)
        m = adapter.PixelMapping(416, 416, 0, 0, 416, 416, 416, 416, 0, 0, 416, 416)
        Q = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
        u = np.array([5, 2, 3])
        transformed = np.array([np.c_[e[:, :3] @ Q.T, e[:, 3]-e[:, :3] @ Q.T @ u] for e in E])
        a = geometry.original_fundamental(E, K, [m, m])
        b = geometry.original_fundamental(transformed, K, [m, m])
        self.assertLess(geometry.fundamental_equivalence(a['F_original'], b['F_original']), 1e-12)

    def test_crop_and_padding_are_invalid_on_inverse(self):
        m = adapter.PixelMapping(1000, 100, 400, 0, 200, 100, 592, 288, 0, 16, 592, 320)
        p = np.array([[400, 0], [500, 50], [599, 99], [399, 0], [600, 50]])
        q, valid = m.forward(p)
        np.testing.assert_array_equal(valid, [True, True, True, False, False])
        back, back_valid = m.inverse(q)
        np.testing.assert_array_equal(back_valid, valid)
        np.testing.assert_allclose(back[valid], p[valid], atol=1e-12)
        self.assertTrue(np.isnan(back[~valid]).all())
        _, valid = m.inverse([[0, 0], [590, 319]])
        self.assertFalse(valid.any())
        # Pixel centers and continuous image edges have different offsets.
        self.assertNotEqual(m.A[0, 2], m.A_edges[0, 2])

    def test_half_pixel_matrix_matches_PIL_bilinear_ramp(self):
        # Bicubic and bilinear use the same pixel-center sampling convention.
        # A float32 ramp measures the convention independently of our matrix.
        image = Image.fromarray(np.tile(np.arange(200, dtype=np.float32), (100, 1)), mode='F')
        resized = np.asarray(image.resize((592, 288), Image.Resampling.BILINEAR))
        m = adapter.PixelMapping(200, 100, 0, 0, 200, 100, 592, 288, 0, 0, 592, 288)
        original, valid = m.inverse([[150, 100], [300, 200]])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(resized[[100, 200], [150, 300]], original[:, 0], atol=1e-5)

    def test_official_preprocessing_pixels_real_shapes_and_alpha(self):
        import torch
        load_fn, _ = adapter.official_api(SOURCE)
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for i, size in enumerate([(1001, 100), (100, 1001), (1919, 1079), (320, 321)]):
                color = (50+i*20, 130, 220, 100)
                p = Path(temp)/f'{i}.png'
                Image.new('RGBA', size, color).save(p); paths.append(str(p))
            for resolution in (416, 512):
                for pair in (paths[:2], paths[2:]):
                    actual, mappings = adapter.preprocess_pair(SOURCE, pair, resolution)
                    reconstructed = []
                    from torchvision.transforms.functional import to_tensor
                    for path, m in zip(pair, mappings):
                        image = load_fn._load_rgb_image(path).crop((m.crop_left, m.crop_top, m.crop_left+m.crop_width, m.crop_top+m.crop_height))
                        t = to_tensor(image.resize((m.resize_width, m.resize_height), Image.Resampling.BICUBIC))
                        t = torch.nn.functional.pad(t, (m.pad_left, m.network_width-m.pad_left-m.resize_width,
                                                       m.pad_top, m.network_height-m.pad_top-m.resize_height), value=1.)
                        reconstructed.append(t)
                    self.assertTrue(torch.equal(actual, torch.stack(reconstructed)))
                    self.assertTrue(all(m.network_width % 16 == m.network_height % 16 == 0 for m in mappings))

    def test_official_camera_encoding_synthetic_round_trip(self):
        import torch
        _, decode = adapter.official_api(SOURCE)
        from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding
        E = torch.zeros(1, 2, 3, 4); E[..., :3] = torch.eye(3)
        E[0, 1, :, 3] = torch.tensor([1., .1, 0])
        K = torch.tensor([[[[500., 0, 312], [0, 480, 176], [0, 0, 1]]]*2])
        encoding = extri_intri_to_pose_encoding(E, K, (352, 624))
        e, k = decode(encoding, (352, 624))
        self.assertEqual(tuple(encoding.shape), (1, 2, 9))
        np.testing.assert_allclose(e.numpy(), E.numpy(), atol=1e-6)
        np.testing.assert_allclose(k.numpy(), K.numpy(), atol=1e-4)

    def test_degenerate_cameras_cannot_become_reliable(self):
        m = adapter.PixelMapping(416, 416, 0, 0, 416, 416, 416, 416, 0, 0, 416, 416)
        E = np.array([np.c_[np.eye(3), [0, 0, 0]]]*2)
        K = np.array([np.diag([400, 400, 1])]*2)
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            geometry.original_fundamental(E, K, [m, m])
        E[1, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'SO'):
            geometry.original_fundamental(E, K, [m, m])
        invalid = geometry.background_validity(None, [], [], [(416, 416)]*2)
        self.assertFalse(invalid['valid']); self.assertEqual(invalid['median_residual'], 'NOT_MEASURED')

    def test_empty_validation_does_not_have_perfect_accuracy(self):
        result = geometry.background_validity(np.eye(3), [], [], [(416, 416)]*2)
        self.assertFalse(result['valid'])
        self.assertEqual(result['inlier_ratio'], 'NOT_MEASURED')


if __name__ == '__main__':
    unittest.main()

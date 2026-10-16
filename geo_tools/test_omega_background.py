"""Reject duplicated support and preserve model-independent fit/eval endpoints."""
import unittest
import numpy as np
from omega_background_audit import distinct_matches


class Tests(unittest.TestCase):
    def test_repeated_orientations_are_one_physical_support_point(self):
        first, second = distinct_matches(np.tile([[12.1, 9.2]], (40, 1)), np.tile([[20.1, 13.2]], (40, 1)))
        self.assertEqual(len(first), 1)

    def test_many_to_one_endpoints_are_not_independent_support(self):
        first = np.array([[10, 10], [20, 20], [30, 30]], float)
        second = np.array([[1, 1], [1, 1], [5, 5]], float)
        a, b = distinct_matches(first, second)
        self.assertEqual(len(a), 2)
        a_reversed, b_reversed = distinct_matches(first[::-1], second[::-1])
        np.testing.assert_array_equal(a, a_reversed)
        np.testing.assert_array_equal(b, b_reversed)
        self.assertEqual(len(set(map(tuple, a))), len(a))
        self.assertEqual(len(set(map(tuple, b))), len(b))

    def test_empty_is_retained_and_corrupt_points_are_refused(self):
        a, b = distinct_matches([], [])
        self.assertEqual(a.shape, (0, 2))
        with self.assertRaises(ValueError):
            distinct_matches([[np.nan, 0]], [[1, 1]])
        with self.assertRaises(ValueError):
            distinct_matches([[1, 1]], [])


if __name__ == '__main__':
    unittest.main()

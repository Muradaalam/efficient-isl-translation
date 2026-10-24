
import unittest
import numpy as np
from pathlib import Path
from src.features.glosl_pose import read_glosl_pose

from src.features.glosl_pose import normalize_glosl_sequence


class TestGloSLNormalization(unittest.TestCase):

    def setUp(self):
        self.pose = np.zeros((3, 75, 3), dtype=np.float32)
        self.mask = np.ones((3, 75), dtype=bool)

        for t, width in enumerate([2.0, 2.0, 0.2]):
            self.pose[t, 11, :2] = [-width / 2, 1]
            self.pose[t, 12, :2] = [width / 2, 1]
            self.pose[t, 23, :2] = [-0.5, 0]
            self.pose[t, 24, :2] = [0.5, 0]
            self.pose[t, 15, :2] = [2, 2]

    def test_output_shape_and_dtype(self):
        result, mask = normalize_glosl_sequence(self.pose, self.mask)

        self.assertEqual(result.shape, (3, 75, 3))
        self.assertEqual(mask.shape, (3, 75))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(mask.dtype, np.bool_)

    def test_uses_median_shoulder_width(self):
        result, _ = normalize_glosl_sequence(self.pose, self.mask)

        # Shoulder widths are [2, 2, 0.2], so median is 2.
        np.testing.assert_allclose(result[2, 15, :2], [1, 1])
        np.testing.assert_allclose(result[0, 15, :2], [1, 1])

    def test_missing_reference_invalidates_frame(self):
        self.mask[1, 11] = False
        result, mask = normalize_glosl_sequence(self.pose, self.mask)

        self.assertFalse(mask[1].any())
        self.assertTrue(np.all(result[1] == 0))

    def test_missing_landmark_remains_zero(self):
        self.mask[0, 15] = False
        result, mask = normalize_glosl_sequence(self.pose, self.mask)

        self.assertFalse(mask[0, 15])
        np.testing.assert_array_equal(result[0, 15], [0, 0, 0])

    def test_all_invalid_references(self):
        self.mask[:, 11] = False
        result, mask = normalize_glosl_sequence(self.pose, self.mask)

        self.assertFalse(mask.any())
        self.assertTrue(np.all(result == 0))

    def test_depth_is_preserved(self):
        self.pose[0, 15, 2] = 0.8
        result, _ = normalize_glosl_sequence(self.pose, self.mask)

        self.assertAlmostEqual(float(result[0, 15, 2]), 0.8)


class TestGloSLReader(unittest.TestCase):

    def test_real_pose_file(self):
        files = sorted(Path("glosl_8000/poses").rglob("*.pose"))

        if not files:
            self.skipTest("GloSL dataset not available")

        pose, mask, confidence = read_glosl_pose(files[0])

        self.assertEqual(pose.ndim, 3)
        self.assertEqual(pose.shape[1:], (75, 3))
        self.assertEqual(mask.shape, pose.shape[:2])
        self.assertEqual(confidence.shape, mask.shape)

        self.assertEqual(pose.dtype, np.float32)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(confidence.dtype, np.float32)

        self.assertTrue(np.isfinite(pose).all())
        self.assertTrue(np.isfinite(confidence).all())
        self.assertTrue(np.all(pose[~mask] == 0))
        self.assertTrue(np.all(confidence[~mask] == 0))
        self.assertTrue(np.all((confidence >= 0) & (confidence <= 1)))


if __name__ == "__main__":
    unittest.main()


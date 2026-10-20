
import unittest
import numpy as np

from src.features.pose_motion import normalize_landmarks


class TestPoseNormalization(unittest.TestCase):

    def setUp(self):
        self.coords = np.zeros((75, 3), dtype=np.float32)
        self.mask = np.zeros(75, dtype=bool)

        self.coords[23, :2] = [0.4, 0.7]
        self.coords[24, :2] = [0.6, 0.7]
        self.coords[11, :2] = [0.4, 0.4]
        self.coords[12, :2] = [0.6, 0.4]

        self.mask[[11, 12, 23, 24]] = True

    def test_output_shapes_and_types(self):
        pose, mask = normalize_landmarks(self.coords, self.mask)

        self.assertEqual(pose.shape, (75, 3))
        self.assertEqual(mask.shape, (75,))
        self.assertEqual(pose.dtype, np.float32)
        self.assertEqual(mask.dtype, np.bool_)

    def test_body_center_and_scale(self):
        pose, mask = normalize_landmarks(self.coords, self.mask)

        hip_center = (pose[23, :2] + pose[24, :2]) / 2
        shoulder_width = np.linalg.norm(pose[11, :2] - pose[12, :2])

        np.testing.assert_allclose(hip_center, [0, 0], atol=1e-6)
        self.assertAlmostEqual(shoulder_width, 1.0, places=5)

    def test_missing_landmarks_remain_zero(self):
        pose, mask = normalize_landmarks(self.coords, self.mask)

        self.assertTrue(np.all(pose[~mask] == 0))

    def test_missing_body_reference(self):
        self.mask[11] = False

        pose, mask = normalize_landmarks(self.coords, self.mask)

        self.assertFalse(mask.any())
        self.assertTrue(np.all(pose == 0))

    def test_zero_shoulder_width(self):
        self.coords[12, :2] = self.coords[11, :2]

        pose, mask = normalize_landmarks(self.coords, self.mask)

        self.assertFalse(mask.any())
        self.assertTrue(np.all(pose == 0))

    def test_depth_is_preserved(self):
        self.coords[11, 2] = 0.25

        pose, mask = normalize_landmarks(self.coords, self.mask)

        self.assertAlmostEqual(pose[11, 2], 0.25, places=6)


if __name__ == "__main__":
    unittest.main()


import unittest
import numpy as np

from src.features.motion import calculate_hand_motion


class TestHandMotion(unittest.TestCase):

    def setUp(self):
        self.pose = np.zeros((3, 75, 3), dtype=np.float32)
        self.mask = np.ones((3, 75), dtype=bool)

    def test_output_shape(self):
        scores = calculate_hand_motion(self.pose, self.mask)

        self.assertEqual(scores.shape, (3,))
        self.assertEqual(scores.dtype, np.float32)

    def test_no_movement(self):
        scores = calculate_hand_motion(self.pose, self.mask)

        np.testing.assert_array_equal(scores, [0, 0, 0])

    def test_hand_movement(self):
        self.pose[1, 33:75, 0] = 2
        self.pose[2, 33:75, 0] = 5

        scores = calculate_hand_motion(self.pose, self.mask)

        np.testing.assert_allclose(scores, [0, 2, 3])

    def test_ignores_body_movement(self):
        self.pose[1, 0:33, 0] = 10

        scores = calculate_hand_motion(self.pose, self.mask)

        np.testing.assert_array_equal(scores, [0, 0, 0])

    def test_missing_landmarks(self):
        self.pose[1, 33:75, 0] = 100
        self.mask[1, 33:75] = False

        scores = calculate_hand_motion(self.pose, self.mask)

        np.testing.assert_array_equal(scores, [0, 0, 0])

    def test_partial_hand_detection(self):
        self.mask[:, 34:75] = False
        self.pose[1, 33, :2] = [3, 4]

        scores = calculate_hand_motion(self.pose, self.mask)

        np.testing.assert_allclose(scores, [0, 5, 5])

    def test_single_frame(self):
        scores = calculate_hand_motion(self.pose[:1], self.mask[:1])

        np.testing.assert_array_equal(scores, [0])

    def test_invalid_input_shape(self):
        with self.assertRaises(ValueError):
            calculate_hand_motion(np.zeros((3, 74, 3)), self.mask)

    def test_returns_valid_pair_counts(self):
        scores, counts = calculate_hand_motion(
            self.pose, self.mask, return_counts=True
        )

        np.testing.assert_array_equal(counts, [0, 42, 42])
        self.assertEqual(counts.dtype, np.int32)
        self.assertEqual(scores.shape, (3,))

    def test_missing_motion_has_zero_coverage(self):
        self.mask[1, 33:75] = False

        scores, counts = calculate_hand_motion(
            self.pose, self.mask, return_counts=True
        )

        np.testing.assert_array_equal(scores, [0, 0, 0])
        np.testing.assert_array_equal(counts, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()


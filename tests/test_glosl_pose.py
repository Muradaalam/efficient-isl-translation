
import unittest
import numpy as np
from pathlib import Path
from src.features.glosl_pose import read_glosl_pose
from pose_format import Pose
from src.features.glosl_pose import FACE_INDICES

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

    
    def test_face_z_centering(self):
        pose = np.zeros((3, 100, 3), dtype=np.float32)
        mask = np.ones((3, 100), dtype=bool)

        pose[:, 11, :2] = [-1, 1]
        pose[:, 12, :2] = [1, 1]
        pose[:, 23, :2] = [-0.5, 0]
        pose[:, 24, :2] = [0.5, 0]

        pose[:, 75:100, 2] = 0.3
        nose_z = np.array([0.1, 0.2, 0.1], dtype=np.float32)

        result, result_mask = normalize_glosl_sequence(
            pose, mask, face_nose_z=nose_z
        )

        np.testing.assert_allclose(result[0, 75:100, 2], 0.2)
        np.testing.assert_allclose(result[1, 75:100, 2], 0.1)
        self.assertTrue(result_mask[:, 75:100].all())

    def test_hand_z_is_wrist_relative(self):
        pose = np.zeros((2, 100, 3), dtype=np.float32)
        mask = np.ones((2, 100), dtype=bool)

        pose[:, 11, :2] = [-1, 1]
        pose[:, 12, :2] = [1, 1]
        pose[:, 23, :2] = [-0.5, 0]
        pose[:, 24, :2] = [0.5, 0]

        pose[:, 33, 2] = 0.2
        pose[:, 34, 2] = 0.5

        pose[:, 54, 2] = -0.1
        pose[:, 55, 2] = 0.3

        result, result_mask = normalize_glosl_sequence(pose, mask)

        np.testing.assert_allclose(result[:, 33, 2], 0, atol=1e-6)
        np.testing.assert_allclose(result[:, 34, 2], 0.3, atol=1e-6)

        np.testing.assert_allclose(result[:, 54, 2], 0, atol=1e-6)
        np.testing.assert_allclose(result[:, 55, 2], 0.4, atol=1e-6)

        self.assertTrue(result_mask.all())



class TestGloSLReader(unittest.TestCase):

    def test_real_pose_file(self):
        files = sorted(Path("glosl_8000/poses").rglob("*.pose"))

        if not files:
            self.skipTest("GloSL dataset not available")

        pose, mask, confidence = read_glosl_pose(files[0])

        self.assertEqual(pose.ndim, 3)
        self.assertEqual(pose.shape[1:], (100, 3))
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

    
    def test_face_landmark_selection(self):
        self.assertEqual(len(FACE_INDICES), 25)
        self.assertEqual(len(set(FACE_INDICES)), 25)
        self.assertTrue(all(0 <= i < 468 for i in FACE_INDICES))

    def test_real_face_landmark_mapping(self):
        files = sorted(Path("glosl_8000/poses").rglob("*.pose"))

        if not files:
            self.skipTest("GloSL dataset not available")

        with files[0].open("rb") as f:
            raw = Pose.read(f.read())

        data = np.ma.filled(raw.body.data, np.nan)[:, 0]
        conf = np.ma.filled(raw.body.confidence, 0)[:, 0]

        converted, mask, _ = read_glosl_pose(files[0])

        # The selected face landmarks occupy indices 75:100.
        # Compare their XY positions after our normalization.
        reference_frames = mask[:, [11, 12, 23, 24]].all(axis=1)
        widths = np.linalg.norm(
            data[:, 11, :2] - data[:, 12, :2], axis=1
        )
        scale = np.median(widths[reference_frames])

        for t in np.where(reference_frames)[0][:10]:
            center = (data[t, 23, :2] + data[t, 24, :2]) / 2

            for j, face_index in enumerate(FACE_INDICES):
                global_index = 33 + face_index

                if conf[t, global_index] <= 0:
                    continue

                expected_xy = (data[t, global_index, :2] - center) / scale

                np.testing.assert_allclose(
                    converted[t, 75 + j, :2],
                    expected_xy,
                    rtol=1e-5,
                    atol=1e-5
                )



if __name__ == "__main__":
    unittest.main()


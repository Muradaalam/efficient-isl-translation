
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.extract_pose_batch import extract_batch


class TestBatchExtraction(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video_dir = self.root / "videos"
        self.output_dir = self.root / "features"

        (self.video_dir / "sentence").mkdir(parents=True)
        (self.video_dir / "sentence" / "video1.mp4").touch()
        (self.video_dir / "sentence" / "video2.MP4").touch()

    def tearDown(self):
        self.temp.cleanup()

    def fake_extract(self, video_path):
        pose = np.ones((5, 75, 3), dtype=np.float32)
        mask = np.ones((5, 75), dtype=bool)
        return pose, mask

    @patch("scripts.extract_pose_batch.extract_pose_from_video")
    def test_saves_features(self, mock_extract):
        mock_extract.side_effect = self.fake_extract

        records = extract_batch(self.video_dir, self.output_dir)

        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["status"] == "success" for r in records))

        saved = list(self.output_dir.rglob("*.npz"))
        self.assertEqual(len(saved), 2)

        with np.load(saved[0]) as data:
            self.assertEqual(data["pose"].shape, (5, 75, 3))
            self.assertEqual(data["valid_mask"].shape, (5, 75))

    @patch("scripts.extract_pose_batch.extract_pose_from_video")
    def test_limit(self, mock_extract):
        mock_extract.side_effect = self.fake_extract

        records = extract_batch(self.video_dir, self.output_dir, limit=1)

        self.assertEqual(len(records), 1)
        self.assertEqual(mock_extract.call_count, 1)

    @patch("scripts.extract_pose_batch.extract_pose_from_video")
    def test_failed_video(self, mock_extract):
        mock_extract.side_effect = ValueError("Unreadable video")

        records = extract_batch(self.video_dir, self.output_dir)

        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["status"] == "failed" for r in records))
        self.assertEqual(len(list(self.output_dir.rglob("*.npz"))), 0)

    @patch("scripts.extract_pose_batch.extract_pose_from_video")
    def test_report_created(self, mock_extract):
        mock_extract.side_effect = self.fake_extract

        extract_batch(self.video_dir, self.output_dir)

        report = self.output_dir / "extraction_report.csv"
        self.assertTrue(report.exists())

        with report.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "success" for row in rows))


if __name__ == "__main__":
    unittest.main()

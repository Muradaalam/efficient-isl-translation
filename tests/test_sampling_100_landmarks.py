
import numpy as np
import pytest

from src.sampling.temporal import sample, calculate_motion_scores


@pytest.mark.parametrize("method", ["uniform", "adaptive"])
@pytest.mark.parametrize("T", [1, 20, 64, 100, 399])
def test_100_landmark_shapes_masks_order(method, T):

    rng = np.random.default_rng(42)

    pose = rng.random((T, 100, 3))
    mask = rng.random((T, 100)) > 0.2

    sampled_pose, sampled_mask, indices = sample(
        pose, mask, 64, method
    )

    n = min(T, 64)

    assert sampled_pose.shape == (64, 100, 3)
    assert sampled_mask.shape == (64, 100)
    assert indices.shape == (64,)

    # All original indices must be unique and chronological
    assert len(np.unique(indices[:n])) == n
    assert np.all(np.diff(indices[:n]) > 0)

    # Check selected pose coordinates and corresponding masks
    np.testing.assert_array_equal(
        sampled_pose[:n],
        pose[indices[:n]]
    )

    np.testing.assert_array_equal(
        sampled_mask[:n],
        mask[indices[:n]]
    )

    # Padding verification
    if T < 64:
        assert np.all(indices[n:] == -1)
        assert np.all(sampled_pose[n:] == 0)
        assert not sampled_mask[n:].any()


def test_motion_ignores_body_and_face():

    pose = np.zeros((3, 100, 3))
    mask = np.ones((3, 100), dtype=bool)

    # Move only body and facial landmarks
    pose[1, :33, 0] = 100
    pose[1, 75:, 0] = 100

    scores = calculate_motion_scores(pose, mask)

    # These movements must not contribute to motion scores
    np.testing.assert_array_equal(scores, [0, 0, 0])

    # Now move the 42 hand landmarks
    pose[1, 33:75, 0] = 2

    scores = calculate_motion_scores(pose, mask)

    np.testing.assert_allclose(scores, [0, 2, 2])


def test_invalid_hand_landmarks_are_ignored():

    pose = np.zeros((2, 100, 3))
    mask = np.ones((2, 100), dtype=bool)

    pose[1, 33, 0] = 100
    pose[1, 34, 0] = 4

    # Invalidate the landmark with the large movement
    mask[1, 33] = False

    scores = calculate_motion_scores(pose, mask)

    assert np.isclose(scores[1], 4 / 41)


def test_adaptive_prioritizes_hand_motion():

    pose = np.zeros((100, 100, 3))
    mask = np.ones((100, 100), dtype=bool)

    # Facial-only movement
    pose[40, 75:, 0] = 100

    # Strong hand movement
    pose[50, 33:75, 0] = 10

    _, _, indices = sample(
        pose, mask, target_frames=64, method="adaptive"
    )

    assert 50 in indices
    assert 51 in indices


def test_empty_and_invalid():

    with pytest.raises(ValueError):
        sample(
            np.zeros((0, 100, 3)),
            np.zeros((0, 100), dtype=bool)
        )

    with pytest.raises(ValueError):
        sample(
            np.zeros((2, 100, 3)),
            np.zeros((2, 99), dtype=bool)
        )

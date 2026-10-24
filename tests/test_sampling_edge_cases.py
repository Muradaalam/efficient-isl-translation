import numpy as np
import pytest

from src.sampling.temporal import sample


@pytest.mark.parametrize("method", ["uniform", "adaptive"])
@pytest.mark.parametrize("T", [1, 20, 64, 100, 300])
def test_different_video_lengths(method, T):

    rng = np.random.default_rng(42)

    pose = rng.random((T, 75, 3))
    mask = np.ones((T, 75), dtype=bool)

    sampled_pose, sampled_mask, indices = sample(
        pose, mask, target_frames=64, method=method
    )

    n = min(T, 64)

    assert sampled_pose.shape == (64, 75, 3)
    assert sampled_mask.shape == (64, 75)
    assert indices.shape == (64,)

    # Check selected frames
    assert len(np.unique(indices[:n])) == n
    assert np.all(np.diff(indices[:n]) > 0)

    # Verify padding for short videos
    if T < 64:
        assert np.all(indices[n:] == -1)
        assert np.all(sampled_pose[n:] == 0)
        assert not np.any(sampled_mask[n:])


@pytest.mark.parametrize("method", ["uniform", "adaptive"])
def test_missing_landmarks(method):

    pose = np.random.default_rng(42).random((100, 75, 3))
    mask = np.ones((100, 75), dtype=bool)

    # Simulate missing landmark detections
    mask[20:40, :] = False

    result, result_mask, indices = sample(
        pose, mask, target_frames=64, method=method
    )

    assert result.shape == (64, 75, 3)
    np.testing.assert_array_equal(
        result_mask, mask[indices]
    )


@pytest.mark.parametrize("method", ["uniform", "adaptive"])
def test_static_video(method):

    # Every frame contains identical coordinates
    pose = np.ones((100, 75, 3))
    mask = np.ones((100, 75), dtype=bool)

    result, result_mask, indices = sample(
        pose, mask, target_frames=64, method=method
    )

    assert len(np.unique(indices)) == 64
    assert np.all(np.diff(indices) > 0)


@pytest.mark.parametrize("method", ["uniform", "adaptive"])
def test_empty_video(method):

    pose = np.empty((0, 75, 3))
    mask = np.empty((0, 75), dtype=bool)

    with pytest.raises(ValueError):
        sample(pose, mask, target_frames=64, method=method)
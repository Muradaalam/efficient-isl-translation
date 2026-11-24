import numpy as np
from src.sampling.temporal import sample


def test_uniform_and_adaptive_sampling():

    rng = np.random.default_rng(42)
    pose = rng.random((300, 75, 3))
    mask = np.ones((300, 75), dtype=bool)

    for method in ["uniform", "adaptive"]:

        sampled_pose, sampled_mask, indices = sample(
            pose,
            mask,
            target_frames=64,
            method=method
        )

        assert sampled_pose.shape == (64, 75, 3)
        assert sampled_mask.shape == (64, 75)
        assert len(np.unique(indices)) == 64
        assert np.all(np.diff(indices) > 0)

        print(f"{method} sampling passed!")
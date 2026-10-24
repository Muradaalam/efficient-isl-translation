
from pathlib import Path
import time
import numpy as np

from src.features.glosl_pose import read_glosl_pose
from src.sampling.temporal import sample

# Load 20 real GloSL pose files
files = sorted(Path("glosl_8000/poses").rglob("*.pose"))[:20]

experiments = [
    ("uniform", 64),
    ("adaptive", 64),
    ("adaptive", 48),
    ("adaptive", 32),
    ("adaptive", 128),
]

for method, budget in experiments:

    times = []
    reductions = []

    for file in files:

        pose, mask, _ = read_glosl_pose(file)

        # Warm up
        sample(pose, mask, budget, method)

        for _ in range(30):
            start = time.perf_counter()
            sampled_pose, sampled_mask, indices = sample(
                pose, mask, budget, method
            )
            times.append(
                (time.perf_counter() - start) * 1000
            )

        # Ignore padded positions
        actual_frames = np.sum(indices >= 0)

        reduction = (
            1 - actual_frames / len(pose)
        ) * 100

        reductions.append(reduction)

    print(
        f"{method:8s} | "
        f"frames={budget:3d} | "
        f"median={np.median(times):.3f} ms | "
        f"mean reduction={np.mean(reductions):.1f}%"
    )

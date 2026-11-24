from pathlib import Path
import time
import numpy as np

from src.features.glosl_pose import read_glosl_pose
from src.sampling.temporal import sample

# Find real GloSL pose files
files = sorted(Path("glosl_8000/poses").rglob("*.pose"))

# Pick 20 evenly distributed files
selected_files = [
    files[i] for i in np.linspace(
        0, len(files) - 1, min(20, len(files)), dtype=int
    )
]

results = {"uniform": [], "adaptive": []}

for file in selected_files:
    pose, mask, _ = read_glosl_pose(file)

    for method in ["uniform", "adaptive"]:
        # Warm-up run
        sample(pose, mask, 64, method)

        times = []

        for _ in range(30):
            start = time.perf_counter()

            sample(pose, mask, 64, method)

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        results[method].append(np.median(times))

for method in results:
    print(
        f"{method}: "
        f"median={np.median(results[method]):.3f} ms, "
        f"mean={np.mean(results[method]):.3f} ms"
    )
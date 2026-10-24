import time
from pathlib import Path

from src.features.glosl_pose import read_glosl_pose
from src.sampling.temporal import sample

pose_file = next(Path("glosl_8000/poses").rglob("*.pose"))
pose, mask, _ = read_glosl_pose(pose_file)

for method in ["uniform", "adaptive"]:
    times = []

    for _ in range(100):
        start = time.perf_counter()
        sample(pose, mask, target_frames=64, method=method)
        times.append((time.perf_counter() - start) * 1000)

    print(f"{method}: {sum(times)/len(times):.3f} ms")
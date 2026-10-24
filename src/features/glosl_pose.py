
from pathlib import Path
import numpy as np
from pose_format import Pose



EXPECTED_COMPONENTS = {
    "POSE_LANDMARKS": 33,
    "FACE_LANDMARKS": 468,
    "LEFT_HAND_LANDMARKS": 21,
    "RIGHT_HAND_LANDMARKS": 21
}


def normalize_glosl_sequence(coordinates, valid_mask, eps=1e-6):
    normalized = np.zeros_like(coordinates, dtype=np.float32)
    output_mask = valid_mask.copy()

    refs = [11, 12, 23, 24]
    usable = valid_mask[:, refs].all(axis=1)

    widths = np.linalg.norm(
        coordinates[:, 11, :2] - coordinates[:, 12, :2], axis=1
    )
    usable &= np.isfinite(widths) & (widths > eps)

    if not usable.any():
        output_mask[:] = False
        return normalized, output_mask

    scale = np.median(widths[usable])
    if not np.isfinite(scale) or scale <= eps:
        output_mask[:] = False
        return normalized, output_mask

    output_mask[~usable] = False

    for t in np.where(usable)[0]:
        center = (coordinates[t, 23, :2] + coordinates[t, 24, :2]) / 2
        valid = output_mask[t]

        normalized[t, valid, :2] = (
            coordinates[t, valid, :2] - center
        ) / scale
        normalized[t, valid, 2] = coordinates[t, valid, 2]

    return normalized, output_mask


def read_glosl_pose(file_path):
    with Path(file_path).open("rb") as file:
        pose_file = Pose.read(file.read())

    components = pose_file.header.components
    offsets = {}
    offset = 0

    for component in components:
        name = component.name
        count = len(component.points)

        if name not in EXPECTED_COMPONENTS:
            raise ValueError(f"Unexpected component: {name}")
        if count != EXPECTED_COMPONENTS[name] or name in offsets:
            raise ValueError(f"Invalid component layout: {name}")

        offsets[name] = np.arange(offset, offset + count)
        offset += count

    if set(offsets) != set(EXPECTED_COMPONENTS):
        raise ValueError("Missing expected pose components")

    indices = np.concatenate([
        offsets["POSE_LANDMARKS"],
        offsets["LEFT_HAND_LANDMARKS"],
        offsets["RIGHT_HAND_LANDMARKS"]
    ])

    # Verify landmark ordering used by our existing MediaPipe pipeline.
    points = {c.name: [str(p) for p in c.points] for c in components}
    if points["POSE_LANDMARKS"][11:13] != ["LEFT_SHOULDER", "RIGHT_SHOULDER"]:
        raise ValueError("Unexpected shoulder landmark ordering")
    if points["POSE_LANDMARKS"][23:25] != ["LEFT_HIP", "RIGHT_HIP"]:
        raise ValueError("Unexpected hip landmark ordering")
    for hand in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
        if points[hand][0] != "WRIST" or points[hand][-1] != "PINKY_TIP":
            raise ValueError(f"Unexpected landmark ordering: {hand}")

    data = np.ma.filled(pose_file.body.data, np.nan)
    conf = np.ma.filled(pose_file.body.confidence, 0)

    if data.ndim != 4 or data.shape[1:] != (1, 543, 3):
        raise ValueError(f"Unexpected pose shape: {data.shape}")
    if conf.shape != data.shape[:3]:
        raise ValueError(f"Unexpected confidence shape: {conf.shape}")

    coordinates = np.asarray(data[:, 0, indices, :], dtype=np.float32)
    confidence = np.asarray(conf[:, 0, indices], dtype=np.float32)

    valid_mask = (
        (confidence > 0)
        & np.isfinite(confidence)
        & np.isfinite(coordinates).all(axis=-1)
    )

    coordinates[~valid_mask] = 0
    confidence[~valid_mask] = 0

    normalized, valid_mask = normalize_glosl_sequence(coordinates, valid_mask)

    confidence[~valid_mask] = 0
    return normalized, valid_mask, confidence

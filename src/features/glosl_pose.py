
from pathlib import Path
import numpy as np
from pose_format import Pose



EXPECTED_COMPONENTS = {
    "POSE_LANDMARKS": 33,
    "FACE_LANDMARKS": 468,
    "LEFT_HAND_LANDMARKS": 21,
    "RIGHT_HAND_LANDMARKS": 21
}


# MediaPipe Face Mesh indices, relative to FACE_LANDMARKS.
FACE_EYEBROWS = [70, 105, 107, 300, 334, 336]
FACE_EYES = [33, 133, 159, 145, 263, 362, 386, 374]
FACE_MOUTH = [61, 291, 0, 17, 13, 14, 78, 308, 81, 311, 87]

FACE_INDICES = FACE_EYEBROWS + FACE_EYES + FACE_MOUTH

assert len(FACE_INDICES) == 25
assert len(set(FACE_INDICES)) == 25




def normalize_glosl_sequence(coordinates, valid_mask, face_nose_z=None, eps=1e-6):
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

        hip_z = (coordinates[t, 23, 2] + coordinates[t, 24, 2]) / 2
        normalized[t, :33, 2] -= hip_z * valid[:33]


        # Normalize hand depth relative to each hand's wrist.
        for start, end in ((33, 54), (54, 75)):
            if valid_mask[t, start]:
                wrist_z = coordinates[t, start, 2]
                normalized[t, start:end, 2] -= wrist_z * output_mask[t, start:end]
            else:
                normalized[t, start:end] = 0
                output_mask[t, start:end] = False


        if face_nose_z is not None and np.isfinite(face_nose_z[t]):
            normalized[t, 75:100, 2] -= face_nose_z[t] * valid[75:100]

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
        offsets["RIGHT_HAND_LANDMARKS"],
        offsets["FACE_LANDMARKS"][FACE_INDICES]
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

    nose_index = offsets["FACE_LANDMARKS"][1]
    face_nose_z = np.asarray(data[:, 0, nose_index, 2], dtype=np.float32)
    face_nose_conf = np.asarray(conf[:, 0, nose_index], dtype=np.float32)

    face_nose_z[(face_nose_conf <= 0) | ~np.isfinite(face_nose_z)] = np.nan

    normalized, valid_mask = normalize_glosl_sequence(
        coordinates, valid_mask, face_nose_z
    )
    missing_nose = ~np.isfinite(face_nose_z)
    normalized[missing_nose, 75:100] = 0
    valid_mask[missing_nose, 75:100] = False
    confidence[~valid_mask] = 0
    return normalized, valid_mask, confidence

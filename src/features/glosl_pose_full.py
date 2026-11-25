import numpy as np
from pose_format import Pose


def read_glosl_pose_full(path):
    with open(path, "rb") as f:
        pose = Pose.read(f.read())

    data = np.ma.asarray(pose.body.data)
    confidence = np.asarray(pose.body.confidence)

    if data.ndim != 4 or data.shape[1] != 1:
        raise ValueError(f"Unexpected pose shape: {data.shape}")

    masked = np.ma.getmaskarray(data[:, 0])
    coords = np.asarray(data[:, 0].filled(0), dtype=np.float32)
    confidence = confidence[:, 0].astype(np.float32)

    components = {}
    offset = 0

    for component in pose.header.components:
        count = len(component.points)
        components[component.name.upper()] = np.arange(offset, offset + count)
        offset += count

    order = []

    for name, count in [
        ("POSE_LANDMARKS", 33),
        ("LEFT_HAND_LANDMARKS", 21),
        ("RIGHT_HAND_LANDMARKS", 21),
        ("FACE_LANDMARKS", 468)
    ]:
        indices = components.get(name)

        if indices is None or len(indices) != count:
            raise ValueError(f"Invalid component: {name}")

        order.extend(indices)

    coords = coords[:, order, :3]
    confidence = confidence[:, order]
    masked = masked[:, order, :3]

    valid = (
        (confidence > 0)
        & np.isfinite(coords).all(axis=-1)
        & ~masked.any(axis=-1)
    )

    coords[~valid] = 0
    coords = normalize_pose(coords, valid)

    return coords, valid, confidence


def normalize_pose(coords, valid):
    coords = coords.copy()

    hip_valid = valid[:, 23] & valid[:, 24]
    shoulder_valid = valid[:, 11] & valid[:, 12]

    hips = (coords[:, 23] + coords[:, 24]) / 2
    shoulders = (coords[:, 11] + coords[:, 12]) / 2

    center = np.zeros_like(hips)
    center[hip_valid] = hips[hip_valid]

    fallback = ~hip_valid & shoulder_valid
    center[fallback] = shoulders[fallback]

    widths = np.linalg.norm(
        coords[:, 11, :2] - coords[:, 12, :2], axis=-1
    )

    usable = shoulder_valid & (widths > 1e-6)
    scale = np.median(widths[usable]) if usable.any() else 1.0

    coords[:, :, :2] -= center[:, None, :2]
    coords[:, :, :2] /= scale

    for start, end, anchor in [
        (0, 33, 23),
        (33, 54, 33),
        (54, 75, 54),
        (75, 543, 75)
    ]:
        depth = coords[:, anchor, 2].copy()
        depth[~valid[:, anchor]] = 0
        coords[:, start:end, 2] -= depth[:, None]
        coords[:, start:end, 2] /= scale

    coords[~valid] = 0

    return coords.astype(np.float32)
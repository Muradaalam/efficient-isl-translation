
import cv2
import mediapipe as mp
import numpy as np


def extract_landmarks(landmark_list, count):
    """
    Extract x, y, z coordinates and a validity mask.

    Args:
        landmark_list: Landmarks detected by MediaPipe.
        count: Expected number of landmarks (33 or 21).

    Returns:
        coordinates: NumPy array of shape (count, 3).
        valid_mask: Boolean array of shape (count,).
    """

    # Initialize all coordinates to zero
    coordinates = np.zeros((count, 3), dtype=np.float32)

    # Initially assume no landmarks were detected
    valid_mask = np.zeros(count, dtype=bool)

    # Process landmarks only if MediaPipe detected them
    if landmark_list is not None:
        for i, landmark in enumerate(landmark_list.landmark[:count]):

            # Store x, y, z coordinates
            coordinates[i] = [
                landmark.x,
                landmark.y,
                landmark.z
            ]

            # Mark this landmark as detected
            valid_mask[i] = True

    return coordinates, valid_mask


def extract_pose_from_video(video_path):
    """
    Extract body and hand landmarks from an entire video.

    Landmark order:
        0-32:  Body pose (33)
        33-53: Left hand (21)
        54-74: Right hand (21)

    Returns:
        poses: NumPy array of shape (T, 75, 3).
        masks: Boolean array of shape (T, 75).

    T is the number of readable video frames.
    """

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))

    # Check whether the video opened successfully
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {video_path}")

    # Store landmarks and masks for every frame
    poses = []
    masks = []

    try:
        # Initialize MediaPipe Holistic
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1
        ) as holistic:

            # Read video frames one by one
            while True:
                success, frame = cap.read()

                # Stop when there are no more readable frames
                if not success:
                    break

                # OpenCV uses BGR; MediaPipe expects RGB
                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                # Detect body and hand landmarks
                results = holistic.process(rgb_frame)

                # Extract 33 body landmarks
                pose, pose_mask = extract_landmarks(
                    results.pose_landmarks,
                    33
                )

                # Extract 21 left-hand landmarks
                left, left_mask = extract_landmarks(
                    results.left_hand_landmarks,
                    21
                )

                # Extract 21 right-hand landmarks
                right, right_mask = extract_landmarks(
                    results.right_hand_landmarks,
                    21
                )

                # Combine all 75 landmarks for this frame
                frame_pose = np.concatenate(
                    [pose, left, right],
                    axis=0
                )

                # Combine the corresponding validity masks
                frame_mask = np.concatenate(
                    [pose_mask, left_mask, right_mask],
                    axis=0
                )

                # Store this frame's data
                poses.append(frame_pose)
                masks.append(frame_mask)

    finally:
        # Always release the video resource
        cap.release()

    # Ensure at least one frame was processed
    if not poses:
        raise ValueError(
            f"No readable frames found in video: {video_path}"
        )

    # Combine all frames into NumPy arrays
    poses = np.stack(poses, axis=0)
    masks = np.stack(masks, axis=0)

    return poses, masks

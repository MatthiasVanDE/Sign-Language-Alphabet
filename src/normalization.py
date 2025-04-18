import math
import numpy as np

def center_landmarks(landmarks, ref_point):
    """
    Moves all landmarks so the reference point (the wrist) is centered on (0,0,0).
    """
    return [(x - ref_point[0], y - ref_point[1], z - ref_point[2]) for (x, y, z) in landmarks]

def scale_landmarks(landmarks, ref_a, ref_b):
    """
    Scale the landmarks relative to the distance between two reference points
    (for example the wrist and the tip of the middle finger).
    """
    distance = math.sqrt(
        (ref_a[0] - ref_b[0])**2 +
        (ref_a[1] - ref_b[1])**2 +
        (ref_a[2] - ref_b[2])**2
    )
    distance = distance if distance != 0 else 1  # prevend dividing by zero
    return [(x / distance, y / distance, z / distance) for (x, y, z) in landmarks]

def extract_and_normalize_landmarks(hand_landmarks):
    """
    Retrieve (x, y, z) coördinates from Mediapipe-handlandmarks, center the wrist
    and scale based on the distance between wrist and the tip of the middlefinger.
    Returns a flattened list of 63 values (21 landmarks)
    """
    raw_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist = raw_landmarks[0]
    middle_finger_tip = raw_landmarks[12]

    centered = center_landmarks(raw_landmarks, wrist)
    scaled = scale_landmarks(centered, wrist, middle_finger_tip)
    return [coord for point in scaled for coord in point]

# --------------------------
# Optionally: Alternative normalisation methods
# --------------------------

def no_normalization(hand_landmarks):
    """
    No translation of scaling, use raw (x, y, z).
    """
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

def translate_only(hand_landmarks):
    """
    Only wrist to (0,0,0); no scaling.
    """
    raw_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist = raw_landmarks[0]
    centered = center_landmarks(raw_landmarks, wrist)
    return [coord for point in centered for coord in point]

def translate_scale(hand_landmarks):
    """
    Translation + scaling.
    """
    raw_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist = raw_landmarks[0]
    middle = raw_landmarks[12]
    centered = center_landmarks(raw_landmarks, wrist)
    scaled = scale_landmarks(centered, wrist, middle)
    return [coord for point in scaled for coord in point]

def rotation_correction(landmarks):
    """
    Rotation correction over the wrist->middlefinger-axis.
    """
    (x,y,z) = landmarks[0] # wrist
    middle = landmarks[12] # might be preprocessed
    target = np.array([x, y+1, z]) # the target is a unit vector from the wristpoint to the middle finger tip

    # Find the rotation axis (cross product)
    crossproduct = np.cross(middle, target)
    axis = crossproduct / np.linalg.norm(crossproduct)

    # Find the rotation angle (dot product)
    cos_angle = np.dot(middle, target)
    sin_angle = np.sin(np.arccos(cos_angle))

    # Use the Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    rotation_matrix = np.eye(3) + K * sin_angle + (1 - cos_angle) * np.dot(K, K)
    return [np.dot(rotation_matrix, landmark) for landmark in landmarks]

def translate_scale_rotate(hand_landmarks):
    """
    Translation + scaling + rotation normalisation.
    """
    raw_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist = raw_landmarks[0]
    middle = raw_landmarks[12]
    centered = center_landmarks(raw_landmarks, wrist)
    scaled = scale_landmarks(centered, wrist, middle)
    rotated = rotation_correction(scaled)
    return [coord for point in rotated for coord in point]

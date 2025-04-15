# utils.py

import math

def center_landmarks(landmarks, ref_point):
    """
    Verschuift alle landmarks zodanig dat het referentiepunt (meestal de pols)
    op (0,0,0) komt te liggen.
    """
    return [(x - ref_point[0], y - ref_point[1], z - ref_point[2]) for (x, y, z) in landmarks]

def scale_landmarks(landmarks, ref_a, ref_b):
    """
    Schaal de landmarks relatief aan de afstand tussen twee referentiepunten
    (bijv. pols en top van middelvinger).
    """
    distance = math.sqrt(
        (ref_a[0] - ref_b[0])**2 +
        (ref_a[1] - ref_b[1])**2 +
        (ref_a[2] - ref_b[2])**2
    )
    distance = distance if distance != 0 else 1  # voorkom delen door nul
    return [(x / distance, y / distance, z / distance) for (x, y, z) in landmarks]

def extract_and_normalize_landmarks(hand_landmarks):
    """
    Haal (x, y, z) coördinaten uit Mediapipe-handlandmarks, centreer rond de pols
    en schaal op basis van de afstand pols-middelvinger.
    Geef een vlakke lijst van 63 waarden terug (21 punten * 3 coördinaten).
    """
    raw_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    # Referentie: pols (index 0) en top middelvinger (index 12)
    wrist = raw_landmarks[0]
    middle_finger_tip = raw_landmarks[12]

    # Centraal zetten en vervolgens schalen
    centered = center_landmarks(raw_landmarks, wrist)
    normalized = scale_landmarks(centered, wrist, middle_finger_tip)

    # Flatten van (x, y, z)-tuples naar één lijst
    return [coord for point in normalized for coord in point]

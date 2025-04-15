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
    Geeft een vlakke lijst van 63 waarden terug.
    """
    raw_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist = raw_landmarks[0]
    middle_finger_tip = raw_landmarks[12]

    centered = center_landmarks(raw_landmarks, wrist)
    scaled = scale_landmarks(centered, wrist, middle_finger_tip)

    return [coord for point in scaled for coord in point]

# --------------------------
# Optioneel: Alternatieve normalisatiemethoden
# --------------------------

def no_normalization(hand_landmarks):
    """
    Geen translatie of schaal, gebruik ruwe (x, y, z).
    """
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

def translation_only(hand_landmarks):
    """
    Alleen pols naar (0,0,0); geen schaling.
    """
    raw_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist = raw_landmarks[0]
    centered = center_landmarks(raw_landmarks, wrist)
    return [coord for point in centered for coord in point]

def translation_scale(hand_landmarks):
    """
    Translatie + schaling (zoals extract_and_normalize_landmarks).
    """
    raw_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    wrist = raw_landmarks[0]
    middle = raw_landmarks[12]
    centered = center_landmarks(raw_landmarks, wrist)
    scaled = scale_landmarks(centered, wrist, middle)
    return [coord for point in scaled for coord in point]

def rotation_correction(hand_landmarks):
    """
    (Placeholder) Rotatiecorrectie langs de pols->middelvinger-as.
    Hier zou je een rotatiematrix/kwaternionberekening kunnen uitvoeren.
    """
    # Voorbeeld: Bepaal pols->middelvinger vector en roteer alle punten
    # zodat deze vector (0, 1, 0) of (1, 0, 0) wordt. Dit vereist extra matrixalgebra.
    pass

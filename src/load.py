import cv2
import mediapipe as mp
import os
import pandas as pd
import json
import math

# === Configuration ===
DATASET_DIR = '../data/train'
ANNOTATIONS_FILE = '../data/_annotations.coco.json'
OUTPUT_CSV = 'hand_landmarks_dataset.csv'

# === Normalization helper functions ===

def center_landmarks(landmarks, ref_point):
    """
    Shift all landmarks so that the reference point (usually the wrist) is at the origin.
    """
    return [(x - ref_point[0], y - ref_point[1], z - ref_point[2]) for (x, y, z) in landmarks]

def scale_landmarks(landmarks, scale_ref_a, scale_ref_b):
    """
    Scale landmarks relative to the distance between two reference points.
    This helps eliminate variations due to hand distance from the camera.
    """
    distance = math.sqrt(
        (scale_ref_a[0] - scale_ref_b[0])**2 +
        (scale_ref_a[1] - scale_ref_b[1])**2 +
        (scale_ref_a[2] - scale_ref_b[2])**2
    )
    distance = distance if distance != 0 else 1  # prevent division by zero
    return [(x / distance, y / distance, z / distance) for (x, y, z) in landmarks]

def extract_and_normalize_landmarks(hand_landmarks):
    """
    Extract (x, y, z) coordinates from Mediapipe landmarks and normalize them
    by centering around the wrist and scaling based on wrist-to-middle-finger distance.
    """
    # Extract raw landmark coordinates
    raw_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    # Reference points
    wrist = raw_landmarks[0]               # Landmark 0: wrist
    middle_finger_tip = raw_landmarks[12]  # Landmark 12: tip of middle finger

    # Center and scale the landmarks
    centered = center_landmarks(raw_landmarks, wrist)
    normalized = scale_landmarks(centered, wrist, middle_finger_tip)

    # Flatten (x, y, z) tuples into a single list of 63 values
    return [coord for point in normalized for coord in point]

# === Load COCO annotations ===
with open(ANNOTATIONS_FILE, 'r') as f:
    coco = json.load(f)

# Map image_id to filename
image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

# Map image_id to class label (e.g., 'A', 'B', etc.)
image_id_to_label = {}
for ann in coco['annotations']:
    image_id = ann['image_id']
    category_id = ann['category_id']
    label_name = next((cat['name'] for cat in coco['categories'] if cat['id'] == category_id), None)
    image_id_to_label[image_id] = label_name

# === Initialize Mediapipe Hands detector ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# === Prepare containers for data ===
all_data = []
all_labels = []

# === Process each annotated image ===
for image_id, filename in image_id_to_filename.items():
    label = image_id_to_label.get(image_id)
    if label is None:
        continue  # Skip if no label is found for this image

    # Load image from disk
    img_path = os.path.join(DATASET_DIR, filename)
    image = cv2.imread(img_path)
    if image is None:
        continue  # Skip if image is not found or unreadable

    # Convert BGR (OpenCV format) to RGB (Mediapipe expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run hand landmark detection
    result = hands.process(image_rgb)

    # If at least one hand is detected
    if result.multi_hand_landmarks:
        # Normalize the landmark coordinates
        normalized_landmarks = extract_and_normalize_landmarks(result.multi_hand_landmarks[0])

        # Save the processed data and corresponding label
        all_data.append(normalized_landmarks)
        all_labels.append(label)

# === Save the result to CSV ===
total_images = len(image_id_to_filename)
successful_samples = len(all_data)

if successful_samples > 0:
    # Create DataFrame with features and labels
    df = pd.DataFrame(all_data)
    df['label'] = all_labels
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset saved as {OUTPUT_CSV} with {successful_samples} samples out of {total_images} total images.")
else:
    print(f"No landmarks were found in the {total_images} annotated images.")
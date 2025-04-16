# load.py

import cv2
import mediapipe as mp
import os
import pandas as pd
import json
import math

# === Nieuw: importeer de herbruikbare functies uit utils.py ===
from utils import extract_and_normalize_landmarks

# === Configuration ===
DATASET_DIR = '../data/train'
ANNOTATIONS_FILE = '../data/_annotations.coco.json'
OUTPUT_CSV = 'hand_landmarks_dataset.csv'

def load_images(to_csv=True, norm_func=extract_and_normalize_landmarks):
    # === Load COCO annotations ===
    with open(ANNOTATIONS_FILE, 'r') as f:
        coco = json.load(f)

    # Map image_id to filename
    image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

    # Map image_id to class label
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

        # Load image from disk
        img_path = os.path.join(DATASET_DIR, filename)
        image = cv2.imread(img_path)

        # Convert BGR (OpenCV format) to RGB (Mediapipe expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run hand landmark detection
        result = hands.process(image_rgb)

        # If at least one hand is detected
        if result.multi_hand_landmarks:
            # Normalize the landmark coordinates
            normalized_landmarks = norm_func(result.multi_hand_landmarks[0])

            # Save the processed data and corresponding label
            all_data.append(normalized_landmarks)
            all_labels.append(label)

    if to_csv:
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
    else:
        return all_data, all_labels

if __name__ == "__main__":
    load_images()

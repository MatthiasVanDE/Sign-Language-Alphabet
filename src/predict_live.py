import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import math

# === Nieuw: importeer de herbruikbare functies uit utils.py ===
from utils import extract_and_normalize_landmarks

# === Configuration ===
MODEL_DIR = '../models'
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')  # Use your preferred model here
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# === Load model and label encoder ===
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# === Initialize Mediapipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# === Start webcam feed ===
cap = cv2.VideoCapture(0)
print("Starting live prediction... Press ESC to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Mirror the image for natural interaction
    frame = cv2.flip(frame, 1)

    # Convert image from BGR to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Normalize the landmarks for prediction
        normalized_input = extract_and_normalize_landmarks(hand_landmarks)

        if len(normalized_input) == 63:
            prediction = model.predict([normalized_input])
            label = label_encoder.inverse_transform(prediction)[0]

            # Draw landmarks and prediction label on the image
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(
                frame,
                f"Prediction: {label}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),  # kleur
                2
            )

    # Show the frame in a window
    cv2.imshow('Sign Language Prediction', frame)

    # Exit on ESC key
    if cv2.waitKey(5) & 0xFF == 27:
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
print("Webcam feed closed.")

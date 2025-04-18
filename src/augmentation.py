import os
import cv2
import numpy as np
import mediapipe as mp

from normalization import extract_and_normalize_landmarks

DATASET_DIR = '../data/train'

def no_aug(X_train, y_train):
    return X_train, y_train

def aug_rotate(X_train, y_train, rotation=15):
    angle_radians = np.radians(rotation)

    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    # Rotate around z-axis
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    original_shape = X_train.shape
    points_2d = X_train.reshape(-1, 3)

    rotated_points = np.dot(points_2d, rotation_matrix.T)
    X_train = np.concatenate((X_train, rotated_points.reshape(original_shape)))
    y_train = np.concatenate((y_train, y_train))
    return X_train, y_train

def aug_flip(X_train, y_train):
    flipped = np.flip(X_train)
    X_train = np.concatenate((X_train, flipped))
    y_train = np.concatenate((y_train, y_train))
    return X_train, y_train

def aug_rotate_and_flip(X_train, y_train):
    X_train, y_train = aug_flip(X_train, y_train)
    X_train, y_train = aug_rotate(X_train, y_train)
    return X_train, y_train


def aug_colors(undetected_img):
    """
    (Placeholder): try to redetect images that are previously undetected using different color schemes
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    remaining_undetected = undetected_img.copy()
    detected_X = []
    detected_Y = []

    color_spaces = ['SRGB', 'GRAY_3CH', 'CLAHE_RGB', 'CLAHE_GRAY_3CH', 'EQUALIZED']
    for c_space in color_spaces:
        still_undetected = []
        for filename, label, _ in remaining_undetected:
            # Load and prepare image
            img_path = os.path.join(DATASET_DIR, filename)
            image = cv2.imread(img_path)
            converted = convert_to_color_space(image, c_space)
             # Run hand landmark detection
            result = hands.process(converted)

            if result.multi_hand_landmarks:
                normalized_landmarks = extract_and_normalize_landmarks(result.multi_hand_landmarks[0])

                detected_X.append(normalized_landmarks)
                detected_Y.append(label)
            else:
                still_undetected.append([filename, label, _])
        newly_detected = len(remaining_undetected) - len(still_undetected)
        print(f"Color_space {c_space}: Detected {newly_detected} new images, {len(still_undetected)} remain")
        remaining_undetected = still_undetected
    return detected_X, detected_Y, remaining_undetected

def aug_bbox(undetected_img): 
    """
    try to redetect images that are previously undetected by using the bounding box + various margins 
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    remaining_undetected = undetected_img.copy()
    detected_X = []
    detected_Y = []
    for margin in range(1, 10):
        still_undetected = []
        for filename, label, bbox in remaining_undetected:
            # Load and prepare image
            img_path = os.path.join(DATASET_DIR, filename)
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # unpack bbox and calculate margins
            [bb_X, bb_Y, bb_W, bb_H] = bbox
            min_margin = 1.0 - (margin/10)
            plus_margin = 1.0 + (margin/10)
            [bb_X, bb_Y, bb_W, bb_H] = [round(bb_X*min_margin), round(bb_Y*min_margin), round(bb_W*plus_margin), round(bb_H*plus_margin)] 
            # crop image to bbox + margin
            bbox_img = image_rgb[bb_Y:bb_Y+bb_H, bb_X:bb_X+bb_W]
             # Run hand landmark detection
            result = hands.process(bbox_img)

            if result.multi_hand_landmarks:
                normalized_landmarks = extract_and_normalize_landmarks(result.multi_hand_landmarks[0])

                detected_X.append(normalized_landmarks)
                detected_Y.append(label)
            else:
                still_undetected.append([filename, label, bbox])

        # Report progress
        newly_detected = len(remaining_undetected) - len(still_undetected)
        print(f"Margin {margin/10:.1f}: Detected {newly_detected} new images, {len(still_undetected)} remain")
        remaining_undetected = still_undetected
    return detected_X, detected_Y, remaining_undetected


def convert_to_color_space(image, color_space):
    # helper function to convert image to color space
    if color_space == 'SRGB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == 'GRAY_3CH':  # Grayscale converted to 3 channels
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif color_space == 'CLAHE_RGB':  # Contrast enhancement then RGB
        # Apply CLAHE to luminance channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    elif color_space == 'CLAHE_GRAY_3CH':  # Contrast enhancement on grayscale, then 3-channel
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    elif color_space == 'EQUALIZED':  # Histogram equalization
        # Convert to YUV and equalize the Y channel
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB)
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

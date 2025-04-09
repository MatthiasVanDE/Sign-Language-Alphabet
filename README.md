# Sign Language Alphabet Recognition

This project focuses on recognizing the American Sign Language (ASL) alphabet using hand landmarks extracted via Mediapipe and classified with traditional machine learning models. It supports all 26 static letters (A–Z) and includes both training and real-time prediction capabilities.

## Project Goal

The objective is to create a pipeline that can:

- Process and normalize static images of hands from an annotated ASL dataset
- Extract hand landmarks using Mediapipe
- Train and evaluate machine learning classifiers (KNN, Random Forest, SVM, Logistic Regression)
- Use a webcam feed for real-time sign recognition
- Provide a modular and extensible framework for future improvements

## Dataset

The dataset consists of images of hands forming ASL letters, annotated in the COCO JSON format (`_annotations.coco.json`). Each annotation includes:

- The image file name
- The label corresponding to the hand gesture (A–Z)
- A single hand per image

Landmarks are extracted using Mediapipe’s `Hands` module with 21 hand keypoints per image (x, y, z).

## Preprocessing and Normalization

To make the dataset model-ready, the following preprocessing steps are applied:

1. **Hand Landmark Extraction**  
   Mediapipe is used to extract 21 hand landmarks (each with x, y, z coordinates) from each image.

2. **Normalization**  
   - **Centering**: All landmarks are centered around the wrist (landmark 0).
   - **Scaling**: The landmarks are scaled by the distance between the wrist and the tip of the middle finger (landmark 12) to eliminate variation in hand size or camera distance.

3. **CSV Output**  
   The processed data is saved into a CSV file where each row contains 63 normalized values (21 landmarks × 3 dimensions) and a label column.

## Model Training

Multiple classifiers are supported, including:

- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression

Training is done using `scikit-learn`. Labels are encoded using `LabelEncoder`, and the data is split using a stratified train/test split. The trained model and label encoder are saved to the `/models` directory using `joblib`.

Model performance is evaluated using:

- Accuracy
- Precision, recall, and F1-score per class (via classification report)
- Optionally, a confusion matrix for detailed analysis

## Real-Time Prediction

A live demo uses OpenCV and Mediapipe to:

- Capture frames from the webcam
- Detect and normalize hand landmarks
- Predict the letter using the trained model
- Display the result on the screen with the annotated hand

The normalization process is fully integrated into the live prediction script, ensuring consistency with the training data.

## Usage

### 1. Preprocess the dataset and extract landmarks

```
python extract_landmarks.py
```

### 2. Train a model

Set the `MODEL_TYPE` variable in `train_model.py` to one of the following:

- `'knn'`
- `'random_forest'`
- `'svm'`
- `'logistic_regression'`

Then run the training script:

```
python train_model.py
```

### 3. Start live prediction

Run the live prediction script with:

```
python live_predict.py
```

Make sure your webcam is connected and accessible.

## Requirements

- Python 3.8 or higher
- OpenCV
- Mediapipe
- scikit-learn
- joblib
- pandas
- numpy

Install all dependencies using:

```
pip install -r requirements.txt
```

## Future Improvements

- Add support for dynamic gestures (e.g., J, Z)
- Include confidence thresholding for low-certainty predictions
- Visualize hand skeletons with keypoint heatmaps
- Train deep learning models (e.g., CNNs or LSTMs)
- Improve dataset with diverse hand shapes and skin tones

## Authors

This project was developed as part of an academic assignment to explore real-time gesture recognition using traditional computer vision and machine learning techniques.

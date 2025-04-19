# Sign Language Alphabet Recognition

This repository demonstrates a pipeline for recognizing static American Sign Language (ASL) alphabet letters using Mediapipe for hand landmark detection and traditional machine learning for classification. It covers the end-to-end process of dataset preparation, model training, and real-time prediction via webcam.

## Overview

The system identifies 21 keypoints on a single hand, then processes these landmarks to reduce variability in position, size, and orientation. These processed features are fed into machine learning models (KNN, Random Forest, SVM, or Logistic Regression) to classify the gesture.

## Folder Structure


- **data/**: Contains images and their annotation file (`_annotations.coco.json`).
- **models/**: Stores trained model files (`.pkl`) and a label encoder.
- **utils.py**: Contains helper functions for normalizing and flattening hand landmarks.
- **load.py**: Reads annotations, applies landmark detection, normalizes them, and saves the feature set to CSV.
- **train_model.py**: Trains one of several supported classifiers (KNN, Random Forest, SVM, or Logistic Regression) on the generated CSV data.
- **predict_live.py**: Launches a webcam-based prediction demo.
- **run_experiments.py**: Contains scripts for comparing multiple models, testing normalization strategies, applying dimensionality reduction, and (optionally) doing cross-user validation.

## Setup

1. **Install Dependencies**  
   Make sure Python 3.8+ is installed, then run: pip install -r requirements.txt

2. **Prepare the Dataset**  
- Place images of hands forming ASL letters in the `data/train/` folder.  
- Update `_annotations.coco.json` accordingly, specifying image filenames and labels.
- During creation of this code, the following dataset was used: https://universe.roboflow.com/pranav-atote-ebc3y/american-sign-language-letters-dtfq9.


3. **Generate CSV Dataset**  
Detect and normalize landmarks for all images: 
  ```
  python load.py
  ```
This produces a CSV file (e.g., `hand_landmarks_dataset.csv`) with the features and labels.

4. **Train a Model**  
- Edit `train_model.py` to set `MODEL_TYPE` to one of:  
  - `knn`  
  - `random_forest`  
  - `svm`  
  - `logistic_regression`
- Then run:
  ```
  python train_model.py
  ```
A trained model file (e.g., `random_forest_model.pkl`) and a `label_encoder.pkl` will be saved under `models/`.

5. **Run Experiments**  
To compare multiple classifiers, try different normalization approaches, visualize via PCA/t-SNE, try different 
augmentation methods or perform cross-user validation:
  ```
  python run_experiments.py
  ```


6. **Live Prediction**  
Start a live demo that detects and classifies hand gestures via webcam:
  ```
  python predict_live.py
  ```
A window will open, showing the recognized letter on screen along with drawn hand landmarks.

## Future Directions

- Extend the dataset for dynamic letters (like J and Z).
- Implement more robust data augmentation (lighting, partial occlusion).
- Explore deeper learning architectures (CNN, LSTM).
- Use automated hyperparameter tuning for the traditional ML models.
- Improve cross-user generalization with more diverse training samples.

## License

This project is provided for educational and research purposes. Refer to third-party libraries and dataset sources for additional license details.


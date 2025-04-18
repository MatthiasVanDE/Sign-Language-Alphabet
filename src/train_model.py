# train_model.py

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from augmentation import aug_rotate_and_flip

# === Configuration ===
INPUT_CSV = 'hand_landmarks_dataset.csv'
MODEL_DIR = '../models'
MODEL_TYPE = 'random_forest'  # 'knn', 'random_forest', 'svm', 'logistic_regression'
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_evaluate(split_dataset, model_type='random_forest'):
    """
    Trains and evaluates one model type. Returns dict with metrics + fitted model.
    """
    X_train, X_test, y_train, y_test = split_dataset
    # Choose model
    if model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=3)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        # Let op: probability=True om later predict_proba te kunnen gebruiken
        model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    # Proba (nodig voor ROC AUC bij binair)
    # Bij multiclass moet je macro-averaging toepassen. Hier simplificeren we even.
    if len(set(y_train)) == 2:  # binair
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    results = {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    return results

if __name__ == "__main__":
    # === Load dataset ===
    df = pd.read_csv(INPUT_CSV)
    X = df.drop(columns=['label']).values
    y_raw = df['label'].values

    # === Encode class labels as integers ===
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # === Train chosen model ===
    print(f"Training model type: {MODEL_TYPE}")
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_aug, y_aug = aug_rotate_and_flip(X_train, y_train)

    metrics = train_and_evaluate([X_aug, X_test, y_aug, y_test], model_type=MODEL_TYPE)
    model = metrics["model"]

    print(f"\nAccuracy on test set: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1-score: {metrics['f1_score']:.2f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.2f}")

    # === Save model and label encoder ===
    joblib.dump(model, os.path.join(MODEL_DIR, f'{MODEL_TYPE}_model.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    print(f"Model saved as '{MODEL_DIR}/{MODEL_TYPE}_model.pkl'")
    print(f"Label encoder saved as '{MODEL_DIR}/label_encoder.pkl'")

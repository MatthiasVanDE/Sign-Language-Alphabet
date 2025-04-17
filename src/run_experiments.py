# run_experiments.py

import pandas as pd
import cv2
import mediapipe as mp
import os
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Importeer de train_and_evaluate-functie
from load import load_images
from train_model import train_and_evaluate

# Importeer normalisatiefuncties
from utils import (
    no_normalization,
    translate_only,
    translate_scale,
    translate_scale_rotate,
)

###############################################################################
# EXPERIMENT 1: Modelvergelijking
###############################################################################

def experiment_model_comparison(X, y):
    """
    Vergelijk 4 modellen: KNN, RandomForest, SVM, LogisticRegression
    op accuracy, precision, recall, f1, (roc).
    """
    models_to_compare = ['knn', 'random_forest', 'svm', 'logistic_regression']
    results = []

    for m in models_to_compare:
        metrics = train_and_evaluate(X, y, model_type=m)
        results.append({
            "model_type": m,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"]
        })

    df_results = pd.DataFrame(results)
    print("\n=== MODEL COMPARISON RESULTS ===")
    print(df_results)
    df_results.to_csv("experiment_model_comparison.csv", index=False)


###############################################################################
# EXPERIMENT 2: Normalisatiestrategieën
###############################################################################

def experiment_normalization_strategies():
    """
    Test various normalisation strategies on the same dataset and compare results.
    """

    print("\n=== Normalization Strategies Experiment ===")
    # 1) none
    # 2) translation
    # 3) translation+scale
    # 4) translation+rotate
    # 5) translation_scale_rotate

    methods = {
        "none": no_normalization,
        "translate_only": translate_only,
        "translate_scale": translate_scale,
        "translate_scale_rotate": translate_scale_rotate
    }

    for name, func in methods.items():
        # Load in the dataset using the different normalization techniques
        print(f"Applying {name} normalization")
        X, y = load_images(to_csv=False, norm_func=func)

        # Train & evaluate
        results = train_and_evaluate(X, y, model_type='random_forest')
        print(f"{name} -> Acc={results['accuracy']:.2f}, F1={results['f1_score']:.2f}")


###############################################################################
# EXPERIMENT 3: Dimensiereductie (PCA, t-SNE)
###############################################################################

def experiment_dimensionality_reduction(X, y):
    """
    Voer PCA en t-SNE uit voor visualisatie in 2D.
    """
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.7)
    plt.title("PCA (2D) Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("pca_visualization.png")
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, alpha=0.7)
    plt.title("t-SNE (2D) Visualization")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.savefig("tsne_visualization.png")
    plt.close()

    print("Dimensionality reduction plots saved: pca_visualization.png and tsne_visualization.png")


###############################################################################
# EXPERIMENT 4: Cross-user validatie
###############################################################################

# Als je in je CSV een kolom 'user_id' hebt, kun je 'GroupKFold' gebruiken:
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

def cross_user_validation(X, y, user_ids, model_type='random_forest'):
    """
    Leave-One-User-Out (LOUO) cross-validation.
    """
    gkf = GroupKFold(n_splits=len(set(user_ids)))
    accuracies = []

    from train_model import train_and_evaluate
    for train_idx, test_idx in gkf.split(X, y, groups=user_ids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train and evaluate
        # Let op: train_and_evaluate doet zelf een train_test_split, dus
        # we moeten daar omheen werken of train_and_evaluate hergebruiken op
        # X_train, y_train/X_test, y_test.
        # Hier doen we 'manual training' voor de eenvoud:
        metrics = train_and_evaluate(X_train, y_train, model_type=model_type)
        model = metrics["model"]

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    return np.mean(accuracies)


###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    # 1) Laad je dataset (normaal, via load.py al gegenereerd)
    df = pd.read_csv("hand_landmarks_dataset.csv")
    X = df.drop(columns=['label']).values
    y_raw = df['label'].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 2) Voer experimenten uit

    # Experiment 1: Normalization strategies
    experiment_normalization_strategies()

    # Experiment 2: Model comparison
    experiment_model_comparison(X, y)

    # Experiment 3: Dimensionality reduction
    experiment_dimensionality_reduction(X, y)

    # Experiment 4: Cross-user (alleen als je user_id in df hebt)
    # if 'user_id' in df.columns:
    #     user_ids = df['user_id'].values
    #     mean_acc = cross_user_validation(X, y, user_ids, model_type='random_forest')
    #     print(f"Cross-user mean accuracy: {mean_acc:.2f}")

    print("\nAll experiments completed.")

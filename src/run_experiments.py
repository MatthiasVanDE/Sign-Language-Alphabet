# run_experiments.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Importeer de train_and_evaluate-functie
from augmentation import aug_bbox, aug_colors, aug_flip, aug_rotate, aug_rotate_and_flip, no_aug
from load import load_images
from train_model import train_and_evaluate

# Importeer normalisatiefuncties
from normalization import (
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
    split_dataset = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for m in models_to_compare:
        metrics = train_and_evaluate(split_dataset, model_type=m)
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

    methods = {
        "none": no_normalization,
        "translate_only": translate_only,
        "translate_scale": translate_scale,
        "translate_scale_rotate": translate_scale_rotate
    }

    for name, func in methods.items():
        # Load in the dataset using the different normalization techniques
        print(f"Applying {name} normalization")
        X, y,_ = load_images(to_csv=False, norm_func=func)

        split_dataset = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train & evaluate
        results = train_and_evaluate(split_dataset, model_type='random_forest')
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
# EXPERIMENT 4: Augmentation strategies
###############################################################################


def experiment_augmentation_strategies(X, y):
    """
    Applies different augmentation strategies both pre-hand-landmarkdetection as pre-training)
    """
    
    # Augment the originally undetected images to try improve detection rates  
    pre_detect_methods = {
        "aug_bbox": aug_bbox,
    }
    X, y, undetected = load_images(to_csv=False)

    for name, func in pre_detect_methods.items():
        X_detected, y_detected, _ = func(undetected)
        X_custom = np.concatenate((X, X_detected))
        y_custom = np.concatenate((y, y_detected))
        split_dataset = train_test_split(
            X_custom, y_custom, test_size=0.2, random_state=42, stratify=y_custom
        )
        results = train_and_evaluate(split_dataset)
        print(f"{name} -> Acc={results['accuracy']:.2f}, F1={results['f1_score']:.2f}")


    # Augment the landmarks to hopefully improve accuracy
    post_detect_methods = {
        "None": no_aug,
        "aug_flip": aug_flip,
        "aug_rotate_and_flip": aug_rotate_and_flip,
    }
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for name, func in post_detect_methods.items():
        print(f"Applying {name} augmentation")
        X_augmented, y_augmented = func(X_train, y_train)
        results = train_and_evaluate([X_augmented, X_test, y_augmented, y_test])
        print(f"{name} -> Acc={results['accuracy']:.2f}, F1={results['f1_score']:.2f}")

###############################################################################
# EXPERIMENT 4B: Rotation augmentation over diffferent angles
###############################################################################

def experiment_rotation_augmentation(X, y):
    """
    Applies different degrees of rotation to the training set and evaluates the accuracy
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rotation_data = []
    for angle in range(-90, 105, 15):
        X_augmented, y_augmented = aug_rotate(X_train, y_train, angle)
        results = train_and_evaluate([X_augmented, X_test, y_augmented, y_test])
        rotation_data.append([angle, results['accuracy'], results['f1_score']])
        print(f"{angle} degrees -> Acc={results['accuracy']:.2f}, F1={results['f1_score']:.2f}")

    # rotation_data = np.array(rotation_data)
    # plt.figure(figsize=(8, 6))
    # plt.plot(rotation_data[:,0], rotation_data[:,1], 'r-o', label="Accuracy")
    # plt.plot(rotation_data[:,0], rotation_data[:,2], 'b-o', label="F1-Score")
    # plt.xlabel('Angle of rotation')
    # plt.ylabel('Score')
    # plt.title('Rotation augmentation results')
    # plt.legend()
    # plt.show()

###############################################################################
# EXPERIMENT 5: Cross-user validatie
###############################################################################

# Als je in je CSV een kolom 'user_id' hebt, kun je 'GroupKFold' gebruiken:
from sklearn.model_selection import GroupKFold, train_test_split
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
        metrics = train_and_evaluate([X_train, y_train, X_test, y_test], model_type=model_type)
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

    # Experiment 4: Augmentation strategies
    experiment_augmentation_strategies(X, y)

    # Experiment 4B: Rotation augmentation
    experiment_rotation_augmentation(X, y)

    # Experiment 5: Cross-user (alleen als je user_id in df hebt)
    # if 'user_id' in df.columns:
    #     user_ids = df['user_id'].values
    #     mean_acc = cross_user_validation(X, y, user_ids, model_type='random_forest')
    #     print(f"Cross-user mean accuracy: {mean_acc:.2f}")

    print("\nAll experiments completed.")

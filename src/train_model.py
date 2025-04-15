import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# === Configuration ===
INPUT_CSV = 'hand_landmarks_dataset.csv'
MODEL_DIR = '../models'
MODEL_TYPE = 'random_forest'  # Options: 'knn', 'random_forest', 'svm', 'logistic_regression'
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load dataset ===
df = pd.read_csv(INPUT_CSV)

# Separate features (X) and labels (y)
X = df.drop(columns=['label']).values
y = df['label'].values

# === Encode class labels as integers ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Split dataset into train and test sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Choose model based on selected MODEL_TYPE ===
if MODEL_TYPE == 'knn':
    print("Training K-Nearest Neighbors classifier...")
    model = KNeighborsClassifier(n_neighbors=3)

elif MODEL_TYPE == 'random_forest':
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)

elif MODEL_TYPE == 'svm':
    print("Training Support Vector Machine classifier...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

elif MODEL_TYPE == 'logistic_regression':
    print("Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')

else:
    raise ValueError("Unsupported MODEL_TYPE. Use 'knn', 'random_forest', 'svm', or 'logistic_regression'.")

# === Train the selected model ===
model.fit(X_train, y_train)

# === Evaluate model performance ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel accuracy on test set: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === Save model and label encoder ===
joblib.dump(model, os.path.join(MODEL_DIR, f'{MODEL_TYPE}_model.pkl'))
joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

print(f"\nModel saved as '{MODEL_DIR}/{MODEL_TYPE}_model.pkl'")
print(f"Label encoder saved as '{MODEL_DIR}/label_encoder.pkl'")
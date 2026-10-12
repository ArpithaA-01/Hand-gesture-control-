import csv
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# File paths
DATA_FILE = os.path.join("dataset", "sign_data.csv")
MODEL_FILE = "sign_model.joblib"

# Check whether the dataset exists
if not os.path.exists(DATA_FILE):
    print("ERROR: Dataset not found!")
    print("Expected file:", DATA_FILE)
    raise SystemExit(1)

# Load data from CSV
X = []
y = []

with open(DATA_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)

    if not reader.fieldnames or "label" not in reader.fieldnames:
        print("ERROR: CSV must contain a 'label' column.")
        raise SystemExit(1)

    feature_columns = [
        column for column in reader.fieldnames
        if column != "label"
    ]

    if len(feature_columns) != 63:
        print("ERROR: Expected 63 hand landmark features.")
        print("Found:", len(feature_columns))
        raise SystemExit(1)

    for row in reader:
        try:
            features = [
                float(row[column])
                for column in feature_columns
            ]

            label = row["label"].strip()

            if label and np.isfinite(features).all():
                X.append(features)
                y.append(label)

        except (ValueError, KeyError, TypeError):
            continue

X = np.asarray(X, dtype=float)
y = np.asarray(y)

if len(X) == 0:
    print("ERROR: No valid training samples found.")
    raise SystemExit(1)

classes, counts = np.unique(y, return_counts=True)

print("\nDataset loaded successfully!")
print("Total samples:", len(X))
print("\nSamples per gesture:")

for label, count in zip(classes, counts):
    print(f"{label}: {count}")

if len(classes) < 2:
    print("ERROR: Collect at least two different gestures.")
    raise SystemExit(1)

# Ensure each class has enough examples for a stratified split
if counts.min() < 2:
    print("ERROR: Each gesture needs at least 2 samples.")
    raise SystemExit(1)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build and train the AI classifier
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=10, probability=True)
)

print("\nTraining AI model...")
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nTraining completed!")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=classes,
        zero_division=0
    )
)

# Save the trained model
joblib.dump(model, MODEL_FILE)

print("\nModel saved successfully!")
print("File:", MODEL_FILE)
print("Recognized classes:", ", ".join(classes))
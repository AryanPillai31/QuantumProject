import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
import joblib
import json
import os
from datetime import datetime
from pymongo import MongoClient

def load_preprocessed_data(base_folder="preprocessed_braintumor_mri"):
    X_train = np.load(os.path.join(base_folder, "train", "X_pca.npy"))
    y_train = np.load(os.path.join(base_folder, "train", "labels.npy"))
    X_test = np.load(os.path.join(base_folder, "test", "X_pca.npy"))
    y_test = np.load(os.path.join(base_folder, "test", "labels.npy"))

    with open(os.path.join(base_folder, "label_map.json"), "r") as f:
        label_map = json.load(f)

    return X_train, y_train, X_test, y_test, label_map

def train_and_store_svm(X_train, y_train, X_test, y_test, label_map):
    clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    print("Training SVM...")
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[label_map[str(i)] for i in range(len(label_map))]))

    # Dual coefficients (alpha*y)
    dual_coefs = clf.dual_coef_[0]  # shape: (n_support_vectors,)
    support_indices = clf.support_
    support_vectors = X_train[support_indices]
    support_labels = y_train[support_indices]
    bias = clf.intercept_[0]

    # Prepare data for MongoDB
    model_data = {
        "model_name": "qsvm-brain-mri-v1",
        "timestamp": datetime.utcnow().isoformat(),
        "feature_dim": X_train.shape[1],
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "num_support_vectors": int(len(support_vectors)),
        "bias": float(bias),
        "support_vectors": [
            {"a_y": float(dual_coefs[i]), "x": support_vectors[i].tolist()}
            for i in range(len(support_vectors))
        ],
        "metrics": {
            "accuracy": float(accuracy),
            "f1_score": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
    }

    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["quantum_svm_db"]
    collection = db["svm_models"]

    # Clear old model if exists
    collection.find_one_and_delete({"model_name": "qsvm-brain-mri-v1"})

    # Insert new model
    insert_result = collection.insert_one(model_data)

    print(f"\nModel saved to MongoDB with _id: {insert_result.inserted_id}")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, label_map = load_preprocessed_data(base_folder="preprocessed_braintumor_mri")
    train_and_store_svm(X_train, y_train, X_test, y_test, label_map)

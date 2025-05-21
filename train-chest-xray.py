import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
import json
import os
from datetime import datetime
from pymongo import MongoClient

from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit.primitives import Sampler

def load_preprocessed_data(base_folder="preprocessed_chestxrays"):
    X_train = np.load(os.path.join(base_folder, "train", "X_pca.npy"))
    y_train = np.load(os.path.join(base_folder, "train", "labels.npy"))
    X_test = np.load(os.path.join(base_folder, "test", "X_pca.npy"))
    y_test = np.load(os.path.join(base_folder, "test", "labels.npy"))

    with open(os.path.join(base_folder, "label_map.json"), "r") as f:
        label_map = json.load(f)

    return X_train, y_train, X_test, y_test, label_map

def train_and_store_qsvm(X_train, y_train, X_test, y_test, label_map):
    algorithm_globals.random_seed = 42
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement="linear")
    quantum_kernel = QuantumKernel(feature_map=feature_map, sampler=Sampler())

    clf = QSVC(quantum_kernel=quantum_kernel)
    print("Training QSVM...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[label_map[str(i)] for i in range(len(label_map))]))

    # QSVC does not expose dual_coefs or support_vectors directly
    model_data = {
        "model_name": "qsvm-chest-xray-v1",
        "timestamp": datetime.utcnow().isoformat(),
        "feature_dim": X_train.shape[1],
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "num_support_vectors": "Not available in QSVC",
        "bias": "Not available in QSVC",
        "support_vectors": "Not available in QSVC",
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
    collection.find_one_and_delete({"model_name": "qsvm-chest-xray-v1"})

    # Insert new model
    insert_result = collection.insert_one(model_data)

    print(f"\nQSVM model saved to MongoDB with _id: {insert_result.inserted_id}")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, label_map = load_preprocessed_data(base_folder="preprocessed_chestxrays")
    train_and_store_qsvm(X_train, y_train, X_test, y_test, label_map)

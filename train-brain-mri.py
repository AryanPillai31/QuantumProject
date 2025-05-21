import numpy as np
import os
import json
from datetime import datetime
from pymongo import MongoClient
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.svm import SVC

def load_preprocessed_data(base_folder="preprocessed_braintumor_mri"):
    X_train = np.load(os.path.join(base_folder, "train", "X_pca.npy"))
    y_train = np.load(os.path.join(base_folder, "train", "labels.npy"))
    X_test = np.load(os.path.join(base_folder, "test", "X_pca.npy"))
    y_test = np.load(os.path.join(base_folder, "test", "labels.npy"))

    with open(os.path.join(base_folder, "label_map.json"), "r") as f:
        label_map = json.load(f)

    return X_train, y_train, X_test, y_test, label_map

def train_and_store_qsvm(X_train, y_train, X_test, y_test, label_map):
    print("Preprocessing data...")

    # Standardize and MinMax Scale
    std_scale = StandardScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)

    combined = np.concatenate([X_train, X_test], axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(combined)
    X_train = minmax_scale.transform(X_train)
    X_test = minmax_scale.transform(X_test)

    # Map labels to -1 and 1 (QSVM requirement)
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    print("Setting up quantum kernel...")
    feature_dim = X_train.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear', insert_barriers=True)
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    print("Computing quantum kernel matrices...")
    kernel_train = qkernel.evaluate(X_train)
    kernel_test = qkernel.evaluate(X_test, X_train)

    print("Training QSVM...")
    clf = SVC(kernel='precomputed')
    clf.fit(kernel_train, y_train)

    y_pred = clf.predict(kernel_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[label_map[str(i)] for i in sorted(label_map.keys())]))

    # QSVM Parameters
    dual_coefs = clf.dual_coef_[0]
    support_indices = clf.support_
    support_vectors = X_train[support_indices]
    support_labels = y_train[support_indices]
    bias = clf.intercept_[0]

    # MongoDB document
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

    print("\nSaving model to MongoDB...")
    client = MongoClient("mongodb://localhost:27017/")
    db = client["quantum_svm_db"]
    collection = db["svm_models"]
    collection.find_one_and_delete({"model_name": "qsvm-brain-mri-v1"})
    insert_result = collection.insert_one(model_data)
    print(f"Model saved to MongoDB with _id: {insert_result.inserted_id}")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, label_map = load_preprocessed_data("preprocessed_braintumor_mri")
    train_and_store_qsvm(X_train, y_train, X_test, y_test, label_map)

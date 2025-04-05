import time
import numpy as np
import os
import json
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from datetime import datetime
from pymongo import MongoClient
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute

# Function to load preprocessed data
def load_preprocessed_data(folder, samples_per_class=15):
    X_pca = np.load(os.path.join(folder, "X_pca.npy"))
    labels = np.load(os.path.join(folder, "labels.npy"))
    
    # Select samples_per_class from each class
    class_0_indices = np.where(labels == 0)[0][:samples_per_class]
    class_1_indices = np.where(labels == 1)[0][:samples_per_class]
    selected_indices = np.concatenate([class_0_indices, class_1_indices])
    
    return X_pca[selected_indices], labels[selected_indices]

dataset_path = "preprocessed_xrays"

samples_size = 40

# Load balanced preprocessed train and test data
X_train, y_train = load_preprocessed_data(f"{dataset_path}/train", samples_per_class=samples_size // 2)
X_test, y_test = load_preprocessed_data(f"{dataset_path}/test", samples_per_class=samples_size // 2)

# Limit to samples_size samples each for quick testing
X_train, y_train = X_train[:samples_size], y_train[:samples_size]
X_test, y_test = X_test[:samples_size], y_test[:samples_size]

# Map labels before training
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

print(f"Using {len(X_train)} training samples and {len(X_test)} test samples.")

# Normalize (Standardization + MinMax Scaling)
std_scale = StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)

samples = np.append(X_train, X_test, axis=0)
minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
X_train = minmax_scale.transform(X_train)
X_test = minmax_scale.transform(X_test)

# Set feature map dimension dynamically based on input size
feature_dim = X_train.shape[1]
map_zz = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear', insert_barriers=True)

# Define a quantum sampler and fidelity
sampler = StatevectorSampler()
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(feature_map=map_zz, fidelity=fidelity)

# Compute Kernel Matrices
print("\nComputing quantum kernel matrices...")
start_time = time.time()

kernel_matrix_train = quantum_kernel.evaluate(X_train)
kernel_matrix_test = quantum_kernel.evaluate(X_test, X_train)

end_time = time.time()
print(f"Quantum Kernel Computation Time: {end_time - start_time:.4f} seconds")

# Train SVM on Quantum Kernel
print("\nTraining SVM classifier...")
svm = SVC(kernel="precomputed")
svm.fit(kernel_matrix_train, y_train)
print("SVM training complete!")

# Predict & Evaluate
y_pred = svm.predict(kernel_matrix_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Get dual coefficients (alpha * y)
dual_coefs = svm.dual_coef_[0]  # shape: (n_support_vectors,)
support_indices = svm.support_  # indices of support vectors in training data
support_vectors = X_train[support_indices]
support_labels = y_train[support_indices]
bias = svm.intercept_[0]

# Prepare data as a list of dictionaries
model_data = {
    "model_name": "qsvm-chest-xray-v1",
    "timestamp": datetime.utcnow().isoformat(),
    "feature_dim": X_train.shape[1],
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "num_support_vectors": len(support_vectors),
    "bias": bias,
    "support_vectors": [
        {"a_y": float(dual_coefs[i]), "x": support_vectors[i].tolist()}
        for i in range(len(support_vectors))
    ],
    "metrics": {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
}

# Connect to MongoDB (adjust URI as needed)
client = MongoClient("mongodb://localhost:27017/")
db = client["quantum_svm_db"]
collection = db["svm_models"]

# Clear old model
collection.find_one_and_delete({"model_name": "qsvm-chest-xray-v1"})

# Insert model data
insert_result = collection.insert_one(model_data)

print(f"Model saved to MongoDB with _id: {insert_result.inserted_id}")
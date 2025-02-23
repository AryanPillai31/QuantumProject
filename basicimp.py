


import numpy as np


from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap


from qiskit_machine_learning.kernels import QuantumKernel





X, y = make_blobs(n_samples=40, centers=2, random_state=42, n_features=2)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)






feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')





quantum_instance = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'))
qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)





kernel_train = qkernel.evaluate(x_vec=X_train)


svm = SVC(kernel='precomputed')
svm.fit(kernel_train, y_train)





kernel_test = qkernel.evaluate(x_vec=X_test, y_vec=X_train)
y_pred = svm.predict(kernel_test)




accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy with Quantum SVM:", accuracy)

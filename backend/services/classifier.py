import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit.circuit.library import ZZFeatureMap
from pymongo import MongoClient

def load_model_from_mongo(model_name):
    """
    Load the latest model entry from MongoDB by model_name.
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client['quantum_svm_db']
    collection = db['svm_models']

    model = collection.find_one({"model_name": model_name}, sort=[("timestamp", -1)])
    if not model:
        raise ValueError(f"Model '{model_name}' not found in DB.")

    return model

def classify(file_path, preprocessor, model_name):
    """
    Classifies the input image using the stored model parameters (classical SVM).
    """
    # Step 1: Preprocess the image
    input_vector = preprocessor.toVector(file_path)

    # Step 2: Load model
    model = load_model_from_mongo(model_name=model_name)

    support_vectors = np.array([sv["x"] for sv in model["support_vectors"]])
    dual_coefs = np.array([sv["a_y"] for sv in model["support_vectors"]])
    bias = model["bias"]

    # Step 3: Compute decision value manually
    dot_products = np.array([np.dot(sv, input_vector.flatten()) for sv in support_vectors])
    decision_value = np.sum(dual_coefs * dot_products) + bias   

    # Step 4: Predict
    prediction = int(np.sign(decision_value))

    return prediction
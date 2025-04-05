import base64
import io
import os
import numpy as np
from flask import Flask, request, send_file, jsonify
from PIL import Image
from modules.zz_feature_map import generate_feature_map_image
import joblib
from preprocessors.digits import DigitPreprocessor
from services import classifier

app = Flask(__name__)

# Create an instance of your model.
# Assume it has already been trained elsewhere and saved or is pre-loaded.
# quantum_svm = QuantumSVMModel()
# Optionally, load the pre-trained model from disk:
# quantum_svm.load("model.pkl")

def apply_pca(X, variance_threshold=0.95):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  
    X_standardized = (X - X_mean) / X_std

    cov_matrix = np.cov(X_standardized, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]  
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    print(f"Optimal number of components: {n_components}")

    top_eigenvectors = eigenvectors[:, :n_components]
    X_pca = np.dot(X_standardized, top_eigenvectors)

    return X_pca, n_components

def extract_features(image_bytes):
    """
    Extract features from the image by applying PCA.
    The image is resized to 64x64 pixels, then each pixel (with its color channels)
    is treated as a sample. PCA is applied to the pixel data, and the mean of the
    PCA-transformed data is used as a feature vector.
    """
    try:
        # Open the image from bytes without converting to grayscale
        image = Image.open(io.BytesIO(image_bytes))
        # Resize the image to a fixed size (e.g., 64x64 pixels)
        image = image.resize((64, 64))
        # Convert image to numpy array (retaining original color channels)
        img_array = np.array(image)
        # Reshape the image to a 2D array: each row is a pixel and columns are color channels
        pixels = img_array.reshape(-1, img_array.shape[-1])
        # Apply PCA to the pixel data
        X_pca, n_components = apply_pca(pixels, variance_threshold=0.95)
        # Compute the mean of the PCA-transformed data across all pixels to form a feature vector
        feature_vector = np.mean(X_pca, axis=0)
        return feature_vector
    except Exception as e:
        raise ValueError("Error processing image: " + str(e))

@app.route("/classify/digits", methods=["POST"])
def classify_route():
    file = request.files["image"]
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Ensure uploads folder exists
    upload_folder = os.path.abspath("uploads")
    os.makedirs(upload_folder, exist_ok=True)

    # Save with absolute path
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    pca = joblib.load(os.path.join("models", "digits", 'pca_4.pkl'))
    std = joblib.load(os.path.join("models", "digits", 'std_4.pkl'))
    minmax = joblib.load(os.path.join("models", "digits", 'minmax_4.pkl'))

    # Instantiate appropriate preprocessor
    preprocessor = DigitPreprocessor(pca, std, minmax)  # load these models as needed

    prediction = classifier.classify(file_path, preprocessor, "qsvm-digits-v1")
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

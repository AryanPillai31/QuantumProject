import base64
import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from model import QuantumSVMModel  # Your pre-trained model file

app = Flask(__name__)

# Create an instance of your model.
# Assume it has already been trained elsewhere and saved or is pre-loaded.
quantum_svm = QuantumSVMModel()
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

@app.route("/classify", methods=["POST"])
def classify():
    """
    API endpoint that accepts an image file, extracts features using PCA,
    and passes them to the model for classification. The model returns both the classification
    and an image of the quantum circuit used during prediction.
    
    Expected form-data key: 'image'
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        # Extract features from the uploaded image using PCA
        features = extract_features(image_bytes)
        # Reshape features to a 2D array (1 sample, N features)
        features = features.reshape(1, -1)

        # Call the model’s API. The method `predict_with_circuit` is assumed to return:
        # (classification, quantum_circuit_image)
        classification, circuit_image = quantum_svm.predict_with_circuit(features)

        # Convert the quantum circuit image (PIL Image) to a base64 encoded string
        buffered = io.BytesIO()
        circuit_image.save(buffered, format="PNG")
        circuit_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Map the classification: 1 -> "yes", -1 -> "no"
        result = "yes" if classification == 1 else "no"

        # Return both the classification and the circuit image
        return jsonify({"classification": result, "circuit_image": circuit_img_str})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

import numpy as np
import os
import cv2
from sklearn.datasets import fetch_olivetti_faces  

def load_custom_images(image_folder, image_size=(128, 128)):
    image_data = []
    
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        img_resized = cv2.resize(img, image_size)  
        img_flattened = img_resized.flatten()  
        image_data.append(img_flattened)

    return np.array(image_data)


def load_builtin_dataset():
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    return faces.data  


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



use_builtin_dataset = True  

if use_builtin_dataset:
    print("Using built-in Olivetti Faces dataset.")
    X_images = load_builtin_dataset()
else:
    print("Using custom X-ray images dataset.")
    X_images = load_custom_images("path/to/xray_images")


X_pca, n_components = apply_pca(X_images)


print("Original shape:", X_images.shape)
print(f"Reduced shape: {X_pca.shape} (using {n_components} components)")

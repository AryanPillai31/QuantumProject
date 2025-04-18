import numpy as np
import os
import cv2
import joblib
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

def load_images(image_folder, image_size=(128, 128), limit_per_label=100):
    image_data = []
    labels = []
    label_map = {}

    print(f"Loading images from {image_folder}...")

    for label_idx, label_name in enumerate(sorted(os.listdir(image_folder))):
        label_path = os.path.join(image_folder, label_name)
        if not os.path.isdir(label_path):
            continue

        label_map[label_idx] = label_name
        file_list = os.listdir(label_path)
        print(f"Processing '{label_name}' ({len(file_list)} images)...")

        count = 0
        for filename in file_list:
            if count >= limit_per_label:
                break

            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping {img_path}, unable to load.")
                continue

            img_resized = cv2.resize(img, image_size)
            img_flattened = img_resized.flatten()

            image_data.append(img_flattened)
            labels.append(label_idx)
            count += 1

    print(f"Loaded {len(image_data)} images from {image_folder}")
    return np.array(image_data), np.array(labels), label_map

def preprocess_and_save_dataset(train_folder, test_folder, output_folder="preprocessed_chestxrays", n_components=100):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "test"), exist_ok=True)

    # Load train and test images
    X_train, y_train, label_map = load_images(train_folder, limit_per_label=100)
    X_test, y_test, _ = load_images(test_folder, limit_per_label=100)

    # === Train preprocessing ===
    print("\nPreprocessing training data...")

    # MinMax Scaling
    minmax_scaler = MinMaxScaler()
    X_train_minmax = minmax_scaler.fit_transform(X_train)

    # Standardization
    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train_minmax)

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_std)

    # Save train data
    np.save(os.path.join(output_folder, "train", "X_pca.npy"), X_train_pca)
    np.save(os.path.join(output_folder, "train", "labels.npy"), y_train)

    # Save transformers
    joblib.dump(minmax_scaler, os.path.join(output_folder, "minmax_scaler.pkl"))
    joblib.dump(std_scaler, os.path.join(output_folder, "std_scaler.pkl"))
    joblib.dump(pca, os.path.join(output_folder, "pca.pkl"))

    with open(os.path.join(output_folder, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=4)

    print("Training preprocessing complete and saved.")

    # === Test preprocessing ===
    print("\nPreprocessing test data...")

    # Important: Use SAME scalers trained on train set!
    X_test_minmax = minmax_scaler.transform(X_test)
    X_test_std = std_scaler.transform(X_test_minmax)
    X_test_pca = pca.transform(X_test_std)

    # Save test data
    np.save(os.path.join(output_folder, "test", "X_pca.npy"), X_test_pca)
    np.save(os.path.join(output_folder, "test", "labels.npy"), y_test)

    print("Test preprocessing complete and saved.")

    print("\nAll preprocessing steps done successfully!")

if __name__ == "__main__":
    train_folder = "chest_xray/train"
    test_folder = "chest_xray/test"
    
    if not os.path.exists(train_folder) or not os.path.exists(test_folder):
        print(f"Train or Test folder not found.")
    else:
        preprocess_and_save_dataset(train_folder, test_folder, output_folder="preprocessed_chestxrays", n_components=100)

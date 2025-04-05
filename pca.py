import numpy as np
import os
import cv2
import json
from sklearn.decomposition import TruncatedSVD

def load_images(image_folder, image_size=(128, 128)):
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

        for filename in file_list:
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
            if img is None:
                print(f"Skipping {img_path}, unable to load.")
                continue
            
            img_resized = cv2.resize(img, image_size)  
            img_flattened = img_resized.flatten()  

            image_data.append(img_flattened)
            labels.append(label_idx)

    print(f"Loaded {len(image_data)} images from {image_folder}")
    return np.array(image_data), np.array(labels), label_map


def apply_svd(X, n_components=10):
    print(f"Applying Truncated SVD to reduce to {n_components} components...")
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X)

    explained = svd.explained_variance_ratio_.sum()
    print(f"Explained variance with {n_components} components: {explained:.4f}")
    
    return X_svd


def save_preprocessed_data(X_svd, labels, label_map, output_folder="preprocessed_data"):
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, "X_pca.npy"), X_svd)
    np.save(os.path.join(output_folder, "labels.npy"), labels)
    
    with open(os.path.join(output_folder, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=4)  

    print(f"Saved preprocessed data in {output_folder}")


# Process train, test, val datasets
paths = ["test", "train", "val"]

for sub_path in paths:
    folder_path = "chest_xray/" + sub_path
    if not os.path.exists(folder_path):
        print(f"Skipping {folder_path}, folder not found.")
        continue

    X_images, y_labels, label_map = load_images(folder_path)
    if len(X_images) == 0:
        print(f"Skipping {folder_path}, no images found.")
        continue

    X_svd = apply_svd(X_images)
    save_preprocessed_data(X_svd, y_labels, label_map, output_folder="preprocessed_xrays/" + sub_path)

print("Preprocessing completed!")

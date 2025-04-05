import os
import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Load only digits 0 and 1
digits = load_digits(n_class=2)
X, y = digits.data, digits.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22, stratify=y)

# Apply PCA to reduce dimensions
n_dim = 4
pca = PCA(n_components=n_dim).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Standardize
scaler = StandardScaler().fit(X_train_pca)
X_train_scaled = scaler.transform(X_train_pca)
X_test_scaled = scaler.transform(X_test_pca)

# Scale to [-1, 1]
all_data = np.vstack([X_train_scaled, X_test_scaled])
minmax = MinMaxScaler(feature_range=(-1, 1)).fit(all_data)
X_train_final = minmax.transform(X_train_scaled)
X_test_final = minmax.transform(X_test_scaled)

# Create directory if it doesn't exist
os.makedirs('backend/models/digits', exist_ok=True)

# Save the preprocessors with dimension suffix
joblib.dump(pca, f'backend/models/digits/pca_{n_dim}.pkl')
joblib.dump(scaler, f'backend/models/digits/std_{n_dim}.pkl')
joblib.dump(minmax, f'backend/models/digits/minmax_{n_dim}.pkl')

# Save to proper structure
base_path = "preprocessed_digits"
os.makedirs(os.path.join(base_path, "train"), exist_ok=True)
os.makedirs(os.path.join(base_path, "test"), exist_ok=True)

np.save(os.path.join(base_path, "train", "X_pca.npy"), X_train_final)
np.save(os.path.join(base_path, "train", "labels.npy"), y_train)
np.save(os.path.join(base_path, "test", "X_pca.npy"), X_test_final)
np.save(os.path.join(base_path, "test", "labels.npy"), y_test)

print("Saved X_pca.npy and labels.npy for train and test splits.")

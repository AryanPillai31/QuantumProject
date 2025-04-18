from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from preprocessors.base import Preprocessor
import numpy as np
import cv2

class BrainMriPreprocessor(Preprocessor):
    def __init__(self, pca_model, std_model, minmax_model):
        self.pca = pca_model
        self.std = std_model
        self.minmax = minmax_model

    def toVector(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.resize(img, (128, 128))
        img = img.flatten()

        vector = self.minmax.transform([img])
        vector = self.std.transform(vector)
        vector = self.pca.transform(vector)

        return vector
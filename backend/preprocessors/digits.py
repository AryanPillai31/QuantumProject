from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from preprocessors.base import Preprocessor
from sklearn.datasets import load_digits
from PIL import Image
import numpy as np

class DigitPreprocessor(Preprocessor):
    def __init__(self, pca_model, std_model, minmax_model):
        self.pca = pca_model
        self.std = std_model
        self.minmax = minmax_model

    def toVector(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("L").resize((8, 8))
        img = np.array(img).flatten()
        vector = self.pca.transform([img])
        vector = self.std.transform(vector)
        vector = self.minmax.transform(vector)
        return vector

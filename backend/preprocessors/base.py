from abc import ABC, abstractmethod
import numpy as np

class Preprocessor(ABC):
    @abstractmethod
    def toVector(self, image_path: str) -> np.ndarray:
        pass

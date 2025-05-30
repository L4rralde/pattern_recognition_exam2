import numpy as np
from PIL import Image

class MyImage:
    def __init__(self, path: str) -> None:
        img = Image.open(path)
        self.img = np.asarray(img)
        self.c = 3

    def flatten(self) -> np.ndarray:
        return self.img.reshape(-1, self.c)

    def from_flatten(self, flatten: np.ndarray) -> None:
        return flatten.reshape(self.img.shape[0], self.img.shape[1], self.c)


class GrayImage(MyImage):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.img = np.round(np.mean(self.img, axis = 2)).astype(int)
        self.c = 1

    def flatten(self) -> np.ndarray:
        return self.img.reshape(-1)
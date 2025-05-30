import numpy as np
import random


def get_median(X: np.ndarray, eps: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    theta = np.mean(X, axis=0)

    for _ in range(max_iter):
        diff = X - theta
        distances = np.linalg.norm(diff, axis=1)

        weights = 1.0 / distances
        numerator = np.sum(X * weights[:, np.newaxis], axis=0)
        denominator = np.sum(weights)
        theta_new = numerator / denominator

        if np.linalg.norm(theta - theta_new) < eps:
            break

        theta = theta_new

    return theta

class MedianKNN:
    def __init__(self, nclasses: int = 2) -> None:
        self.nclasses = nclasses
        self.medians = None

    def group(self, X: np.ndarray, medians: list) -> np.ndarray:
        clusters = {i: [] for i in range(self.nclasses)}

        for x in X:
            dists = [
                np.linalg.norm(x - median)
                for median in medians
            ]
            clusters[np.argmin(dists)].append(x)
        
        return {
            i: np.array(cluster)
            for i, cluster in clusters.items()
        }

    def fit(self, X: np.ndarray, **kwargs) -> "MedianKNN":
        rng = np.random.default_rng(**kwargs)  # Optional seed for reproducibility
        indices = rng.choice(X.shape[0], size=self.nclasses, replace=False)

        # Get the selected rows
        medians = X[indices]
        
        for _ in range(100):
            clusters = self.group(X, medians)
            medians = [
                get_median(cluster)
                for cluster in clusters.values()
            ]

        self.medians = medians
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.medians[np.newaxis, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        new_X = []
        for x in X:
            dists = [
                np.linalg.norm(x - median)
                for median in self.medians
            ]
            new_X.append(self.medians[np.argmin(dists)])
        return np.array(new_X)

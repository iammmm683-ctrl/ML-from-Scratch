import numpy as np

class KNearestNeighbor:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.astype(int)

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid num_loops value")
        return self.predict_labels(dists, k)

    def compute_distances_two_loops(self, X):
        num_test, num_train = X.shape[0], self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.linalg.norm(X[i] - self.X_train[j])
        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        dists = np.zeros((num_test, self.X_train.shape[0]))
        for i in range(num_test):
            dists[i] = np.linalg.norm(self.X_train - X[i], axis=1)
        return dists

    def compute_distances_no_loops(self, X):
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        T_sq = np.sum(self.X_train**2, axis=1)
        return np.sqrt(np.maximum(X_sq + T_sq - 2 * X @ self.X_train.T, 0))

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test, int)
        for i in range(num_test):
            nearest = np.argsort(dists[i])[:k]
            votes = np.bincount(self.y_train[nearest])
            y_pred[i] = np.argmax(votes)
        return y_pred

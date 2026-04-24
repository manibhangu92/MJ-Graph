import numpy as np
from sklearn.cluster import KMeans
from Dependencies.utility import euclidean_distance, minkowski_distance, cosine_distance

# ======================================================
# General Metric Tree
# ======================================================
class Tree_G:
    def __init__(self, metric="euclidean", p=2):
        """
        metric : 'euclidean' | 'minkowski' | 'cosine'
        p      : order of Minkowski distance (only if metric='minkowski')
        """
        self.metric = metric
        self.p = p
        self.X = None
        self.root = None

    # --------------------------------------------------
    # Distance function (general space)
    # --------------------------------------------------
    def dist(self, x, y):
        if self.metric == "euclidean":
            return euclidean_distance(x, y);
        elif self.metric == "minkowski":
            return minkowski_distance(x, y, self.p);
        elif self.metric == "cosine":
            return cosine_distance(x, y);
        else:
            raise ValueError("Unknown metric")

    # --------------------------------------------------
    # Center computation
    # --------------------------------------------------
    def compute_center(self, indices):
        center = np.mean(self.X[indices], axis=0)

        # Normalize centroid for cosine space
        if self.metric == "cosine":
            norm = np.linalg.norm(center)
            if norm != 0:
                center = center / norm
        return center

    # --------------------------------------------------
    # Recursive tree generation
    # --------------------------------------------------
    def generate_tree(self, indices, center, depth=0):

        # Leaf node
        if len(indices) == 1:
            return {
                "indices": indices,
                "center": center,
                "left": None,
                "right": None
            }

        # Two-point split
        elif len(indices) == 2:
            i1, i2 = indices
            c1 = self.X[i1]
            c2 = self.X[i2]

            if self.metric == "cosine":
                c1 = c1 / np.linalg.norm(c1)
                c2 = c2 / np.linalg.norm(c2)

            return {
                "center": center,
                "indices": None,
                "left": self.generate_tree(np.array([i1]), c1, depth + 1),
                "right": self.generate_tree(np.array([i2]), c2, depth + 1)
            }

        # Recursive split
        else:
            # KMeans is Euclidean, used only as a partition heuristic
            kmeans = KMeans(n_clusters=2, init="random", n_init=5)
            kmeans.fit(self.X[indices])

            idx_C1 = indices[kmeans.labels_ == 0]
            idx_C2 = indices[kmeans.labels_ == 1]

            # Fallback split (degenerate clustering)
            if len(idx_C1) == 0 or len(idx_C2) == 0:
                half = len(indices) // 2
                idx_C1 = indices[:half]
                idx_C2 = indices[half:]

            center_C1 = self.compute_center(idx_C1)
            center_C2 = self.compute_center(idx_C2)

            return {
                "center": center,
                "indices": None,
                "left": self.generate_tree(idx_C1, center_C1, depth + 1),
                "right": self.generate_tree(idx_C2, center_C2, depth + 1)
            }

    # --------------------------------------------------
    # Fit tree
    # --------------------------------------------------
    def fit(self, X):
        self.X = X.copy()

        # Normalize data for cosine space
        if self.metric == "cosine":
            self.X = self.X / np.linalg.norm(self.X, axis=1, keepdims=True)

        root_center = self.compute_center(np.arange(len(self.X)))
        self.root = self.generate_tree(np.arange(len(self.X)), root_center)

    # --------------------------------------------------
    # Greedy nearest neighbor search
    # --------------------------------------------------
    def find_nearest_neighbor(self, test_point, root):

        if root["left"] is None and root["right"] is None:
            return root["indices"], 0

        d_left = self.dist(test_point, root["left"]["center"])
        d_right = self.dist(test_point, root["right"]["center"])

        if d_left <= d_right:
            idx, visited = self.find_nearest_neighbor(test_point, root["left"])
        else:
            idx, visited = self.find_nearest_neighbor(test_point, root["right"])

        return idx, visited + 1

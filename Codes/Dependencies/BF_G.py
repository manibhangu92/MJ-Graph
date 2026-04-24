from Dependencies.utility import euclidean_distance, minkowski_distance, cosine_distance

# ======================================================
# Brute Force k-NN (General Metric Space)
# ======================================================

def calculate_distance(a, b, metric = "euclidean", p = 2):
    if metric == "euclidean":
        return euclidean_distance(a, b);
    elif metric == "minkowski":
        return minkowski_distance(a, b, p);
    elif metric == "cosine":
        return cosine_distance(a, b);
    else:
        raise ValueError("Unknown metric")

class BF_G:
    def __init__(self, metric = "euclidean", p = 2):
        self.metric = metric;
        self.p = p;
        
    def query(self, query_point, X, n_neighbors = 5):
        distances = []
        for idx in range(len(X)):
            d = calculate_distance(query_point, X[idx], self.metric, self.p);
            distances.append((d, idx));
        distances.sort(key=lambda t: t[0]);
        neighbors_idx = [];
        for i in range(n_neighbors):
            neighbors_idx.append(distances[i][1])
        return neighbors_idx
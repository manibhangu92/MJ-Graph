from Dependencies.utility import euclidean_distance

class BF:
    def query(self, query_point, X, n_neighbors = 5):
        distances = []
        for idx in range(len(X)):
            d = euclidean_distance(query_point, X[idx]);
            distances.append((d, idx));
        distances.sort(key=lambda t: t[0]);
        neighbors_idx = [];
        for i in range(n_neighbors):
            neighbors_idx.append(distances[i][1])
        return neighbors_idx

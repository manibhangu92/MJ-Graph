import numpy as np
from joblib import Parallel, delayed
from Dependencies.utility import euclidean_distance, minkowski_distance, cosine_distance
from Dependencies.Q import Q
from Dependencies.Tree_G import Tree_G

# ======================================================
# Graph Construction (Metric Dominance)
# ======================================================
def _process_single_point(i, X, metric, p):
    n = X.shape[0]
    row = np.zeros(n, dtype=bool)

    def dist(a, b):
        if metric == "euclidean":
            return euclidean_distance(a, b)
        elif metric == "minkowski":
            return minkowski_distance(a, b, p)
        elif metric == "cosine":
            return cosine_distance(a, b)

    dists = np.array([dist(X[i], X[j]) for j in range(n)])
    order = np.argsort(dists)

    selected = []

    for idx in order:
        if idx == i:
            continue

        dij = dists[idx]
        violated = False

        for k in selected:
            if dist(X[k], X[idx]) <= dij:
                violated = True
                break

        if not violated:
            row[idx] = True
            selected.append(idx)

    return i, row


# ======================================================
# AMJG-PE (General Metric Version)
# ======================================================
class AMJG_PEG:
    def __init__(self, metric="euclidean", p=2, n_jobs=-1):
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs
        self.additional = 3
        self.A = None
        self.T = None

    def dist(self, x, y):
        if self.metric == "euclidean":
            return euclidean_distance(x, y)
        elif self.metric == "minkowski":
            return minkowski_distance(x, y, self.p)
        elif self.metric == "cosine":
            return cosine_distance(x, y)

    def create_adjancency_matrix(self, X):
        n = X.shape[0]
        A = np.zeros((n, n), dtype=bool)

        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(_process_single_point)(i, X, self.metric, self.p)
            for i in range(n)
        )

        for i, row in results:
            A[i] = row

        return A

    def construct_graph(self, X):
        print("Constructing AMJG-PE graph...")
        self.A = self.create_adjancency_matrix(X)
        self.T = Tree_G()
        print("Constructing AMJG-PE graph...")
        self.T.fit(X)
        print("Done...");

    # --------------------------------------------------
    # KNN Search
    # --------------------------------------------------
    # ---------------- Neighbor Queue --------------------
    def add_neighbor(self, u, u_dist, k):
        is_discarded = False;
        if self.neighbor_queue.size() < k:
            self.neighbor_queue.push(u_dist, u)
            return True, is_discarded;
        elif u_dist <= self.neighbor_queue.peek()[0]:
            self.neighbor_queue.pop()
            self.neighbor_queue.push(u_dist, u)
            return False, is_discarded;
        else:
            is_discarded = True;
            return False, is_discarded;

    # ---------------- KNN Search ------------------------
    def search_neighbors(self, query_point, start, X, k):
        live_queue = Q(is_min=True);
        visited = np.zeros(X.shape[0]);
        
        dist_start =  self.dist(query_point, X[start]);
        visited[start] = 1;
        live_queue.push(dist_start, start);
        search_radius = dist_start;
        while(not live_queue.is_empty()):
            #extract element
            dist_u, u = live_queue.pop();
            visited[u] = 2;
            response, is_discarded = self.add_neighbor(u, dist_u, k);
            search_radius = self.neighbor_queue.peek()[0];
            if(not response):
                #process neighbors
                v = np.where(self.A[u])[0];
                for idx in v:
                    if(visited[idx] == 0 and not is_discarded):
                        d = self.dist(query_point, X[idx]);
                        if(d <= search_radius):
                            live_queue.push(d, idx);
                        visited[idx] = 1;
            else:
                v = np.where(self.A[u])[0];
                for idx in v:
                    if(visited[idx] == 0):
                        d = self.dist(query_point, X[idx]);
                        live_queue.push(d, idx);
                        visited[idx] = 1;
        return len(np.where(visited > 0)[0]);

    # --------------------------------------------------
    # Query
    # --------------------------------------------------
    def query(self, query, X, n_neighbors=5, n_additional = 3):
        self.additional = n_additional;
        self.neighbor_queue = Q(is_min=False)
        start_idx, tree_visited = self.T.find_nearest_neighbor(query, self.T.root)
        #nearest_vertex, reach_visited = self.reach_nearest_vertex(query_point, start_vertex, X)
        search_visited = self.search_neighbors(query, start_idx[0], X, n_neighbors + self.additional)
        idxs = np.array([v for k, v in self.neighbor_queue.heap])
        dists = np.abs(np.array([k for k, v in self.neighbor_queue.heap]))
        neighbors = idxs[np.argsort(dists)][:n_neighbors]
        return neighbors, search_visited + tree_visited, 0

    # --------------------------------------------------
    # Graph Stats
    # --------------------------------------------------
    def find_degree(self):
        degrees = np.sum(self.A, axis=1)
        return np.mean(degrees), np.max(degrees)

    def find_total_edges(self):
        return int(np.sum(self.A) / 2)


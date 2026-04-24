import numpy as np

from Dependencies.Q import Q
from Dependencies.Tree import Tree
from Dependencies.utility import euclidean_distance
"""
Approximate Manpreet Singh & Jitender Kumar Chhabra Graph

Min-Heap based Sireal Implementation
O(n^2.log_2(n)) construction time

"""
class AMJG_E():
    def __init__(self):
        self.A = None;
        self.T = None;
        self.neighbor_queue = Q(is_min=False);
        self.level = 3;
        
    def check_circle_constraint(self, X, center, R, selected_idx):
        diff = X[selected_idx] - center;
        sq_sum = np.sum(diff * diff, axis=1)
        return True if len(np.where(sq_sum <= R * R)[0]) > 0 else False;

    def distance_min_heap(self, X, i):
        n = X.shape[0]
        ref_point = X[i]
        heap = Q(is_min=True);
        for j in range(n):
            if j != i:
                dist = np.linalg.norm(X[j] - ref_point);
                heap.push(dist, j);
        return heap

    def create_adjancency_matrix(self, X, n):
        A = np.zeros((n, n), dtype = bool);
        for i in range(n):
            heap = self.distance_min_heap(X, i);
            selected_neighbors = set();
            while(not heap.is_empty()):
                d, idx = heap.pop();
                if(not self.check_circle_constraint(X, X[idx], d, list(selected_neighbors))):
                    A[i][idx] = 1;
                    selected_neighbors.add(idx);
        return A;   
    
    """Graph construction"""
    def construct_graph(self, X):
        n, d = X.shape;
        print("Creating adjecancy matrix...");
        self.A = self.create_adjancency_matrix(X, n);
        self.T = Tree();
        self.T.fit(X);
        return;
        
    # ---------------- Greedy Reach ----------------------
    def reach_nearest_vertex(self, query_point, start_vertex, X):
        live_queue = Q(is_min=True)
        visited = np.zeros(X.shape[0])
        nearest_vertex = None
        dist_nearest = None

        dist_start = euclidean_distance(query_point, X[start_vertex])
        visited[start_vertex] = 1
        live_queue.push(dist_start, start_vertex)

        while not live_queue.is_empty():
            dist_u, u = live_queue.pop()
            visited[u] = 2

            if nearest_vertex is None or dist_nearest >= dist_u:
                nearest_vertex, dist_nearest = u, dist_u

            v = np.where(self.A[u])[0]
            dist = np.array([euclidean_distance(query_point, X[idx]) for idx in v])

            for d, idx in zip(dist, v):
                if d <= dist_u and visited[idx] == 0:
                    live_queue.push(d, idx)

            visited[v] = 1
        return nearest_vertex, len(np.where(visited > 0)[0])

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
        
        dist_start = euclidean_distance(query_point, X[start]);
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
                        d = euclidean_distance(query_point, X[idx]);
                        if(d <= search_radius):
                            live_queue.push(d, idx);
                        visited[idx] = 1;
            else:
                v = np.where(self.A[u])[0];
                for idx in v:
                    if(visited[idx] == 0):
                        d = euclidean_distance(query_point, X[idx]);
                        live_queue.push(d, idx);
                        visited[idx] = 1;
        return len(np.where(visited > 0)[0]);

    
    def l3_search(self, nearest_vertex, X, level = 0):
        idxs = set([nearest_vertex]);
        if(level == self.level):
            return idxs;
        for n_idx in np.where(self.A[nearest_vertex] == 1)[0]:
            idxs |= self.l3_search(n_idx, X, level+1);
        return idxs;
        

    # ---------------- Query -----------------------------
    def query(self, query_point, X, n_neighbors=5):
        reach_visited = 0;
        self.neighbor_queue = Q(is_min=False);
        start_vertex = self.T.find_nearest_neighbor(query_point, self.T.root)[0]
        #nearest_vertex, reach_visited = self.reach_nearest_vertex(query_point, start_vertex, X)
        search_visited = self.search_neighbors(query_point, start_vertex, X, n_neighbors * self.mul);
        idxs = np.array([v for k, v in self.neighbor_queue.heap]);
        dist = np.abs(np.array([k for k, v in self.neighbor_queue.heap]));
        neighbors = idxs[np.argsort(dist)][0:n_neighbors];
        return neighbors, search_visited, reach_visited

    # ---------------- Reachability Test -----------------
    def test_reachability(self, X, s_idx, d_idx):
        start = s_idx
        while start != d_idx:
            neighbor_idxs = list(np.where(self.A[start])[0])
            neighbor_idxs.append(start)

            min_distance_idx = np.argmin(
                np.linalg.norm(X[neighbor_idxs] - X[d_idx], axis=1)
            )

            if start == neighbor_idxs[min_distance_idx]:
                return False
            start = neighbor_idxs[min_distance_idx]

        return True

    # ---------------- Degree ----------------------------
    def find_degree(self):
        degrees = np.sum(self.A, axis=1)
        return np.mean(degrees), np.max(degrees)

    # ---------------- Total Edges -----------------------
    def find_total_edges(self):
        return np.sum(self.A) / 2

import numpy as np

from Dependencies.Q import Q
from Dependencies.Tree import Tree
from Dependencies.utility import euclidean_distance
"""
Approximate Manpreet Singh & Jitender Kumar Chhabra Graph

O(n^3) nearest neighbor based implementation

"""
class AMJG():
    def __init__(self):
        self.A = None;
        self.T = None;
        self.neighbor_queue = Q(is_min=False);
        
    def create_distance_matrix(self, X):
        sq_norms = np.sum(X**2, axis=1).reshape(-1, 1);
        D = np.sqrt(sq_norms + sq_norms.T - 2 * X @ X.T);
        return D;

    def create_adjancency_matrix(self, D):
        A = np.zeros(D.shape, dtype = bool);
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if(i != j):
                    candidates = np.where(
                        (D[:, j] <= D[i][j]) & 
                        (np.arange(len(D[:, j])) != i) &
                        (np.arange(len(D[:, j])) != j)
                                )[0];
                    if(len(candidates) == 0):
                        A[i][j] = True;
                    else:
                        A[i][candidates[np.argmin(D[i][candidates])]] = True;
        return A;   
    
    """Graph construction"""
    def construct_graph(self, X):
        n, d = X.shape;
        print("Creating distance matrix...");
        D = self.create_distance_matrix(X);
        print("Creating adjecancy matrix...");
        self.A = self.create_adjancency_matrix(D);
        self.T = Tree();
        self.T.fit(X);
        return;
        
    def reach_nearest_vertex(self, query_point, start_vertex, X):
        live_queue = Q(is_min=True);
        visited = np.zeros(X.shape[0]);
        nearest_vertex = None;
        dist_nearest = None;
        
        dist_start = euclidean_distance(query_point, X[start_vertex]);
        visited[start_vertex] = 1;
        live_queue.push(dist_start, start_vertex);
        
        while(not live_queue.is_empty()):
            #extract element
            dist_u, u = live_queue.pop();
            visited[u] = 2;
            if nearest_vertex == None or dist_nearest >= dist_u:
                nearest_vertex, dist_nearest = u, dist_u
            
            #process neighbors
            v = np.where(self.A[u])[0];
            dist = np.array([euclidean_distance(query_point, X[idx]) for idx in v]);
            for d, idx in zip(dist, v):
                if d <= dist_u and visited[idx] == 0:
                    live_queue.push(d, idx);
            visited[v] = 1;
        return nearest_vertex, len(np.where(visited > 0)[0]);
    
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
        
    
    def query(self, query_point, X, n_neighbors = 5):
        n, d = X.shape;
        self.neighbor_queue = Q(is_min=False);
        start_vertex = self.T.find_nearest_neighbor(query_point, self.T.root)[0];
        nearest_vertex, reach_visited = self.reach_nearest_vertex(query_point, start_vertex, X);
        search_visited = self.search_neighbors(query_point, nearest_vertex, X, n_neighbors);
        return np.array([v for k, v in self.neighbor_queue.heap]), search_visited, reach_visited; 
    
    """
        Function to test the reachability of destination vertex from the start vertex
        based on greedy routing.
    """    
    def test_reachability(self, X, s_idx, d_idx):
        start = s_idx;
        while(start != d_idx):
            neighbor_idxs = list(np.where(np.array(self.A[start]))[0]);
            neighbor_idxs.append(start);
            min_distance_idx = np.argmin(np.linalg.norm(X[neighbor_idxs] - X[d_idx], axis = 1));
            if(start == neighbor_idxs[min_distance_idx]):
                return False;
            else:
                start = neighbor_idxs[min_distance_idx];
        return True;

    """
        Function to find the average and maximum degree of the node in the graph.
    """ 
    def find_degree(self):
        degrees = np.sum(self.A, axis=1);
        return np.mean(degrees), np.max(degrees);
    
    """Total edges in the graph"""
    def find_total_edges(self):
        return np.sum(self.A) / 2;
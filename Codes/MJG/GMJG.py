import numpy as np
from collections import Counter

"""
Manpreet Singh & Jitender Kumar Chhabra Graph (MJ-Graph)

Greedy solution of set-hitting method based implementation of MJ-Graph
Greedy Lower Bound because Exact Lower Bound is NP-Complete (infeasible)
O(n^4)

"""
class GMJG():
    def __init__(self):
        self.A = None;
        
    def create_distance_matrix(self, X):
        sq_norms = np.sum(X**2, axis=1).reshape(-1, 1);
        D = np.sqrt(sq_norms + sq_norms.T - 2 * X @ X.T);
        return D;

    def create_candidate_matrix(self, D):
        C = [[set() for j in range(D.shape[1])] for i in range(D.shape[0])];
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if(i != j):
                    candidates_i = np.where(
                        (D[:, j] <= D[i][j]) & 
                        (np.arange(len(D[:, j])) != i) &
                        (np.arange(len(D[:, j])) != j)
                                )[0];
                    candidates_j = np.where(
                        (D[:, i] <= D[i][j]) & 
                        (np.arange(len(D[:, j])) != i) &
                        (np.arange(len(D[:, j])) != j)
                                )[0];
                    C[i][j] = C[j][i] = set(candidates_i) & set(candidates_j);   
        return C;
    
    def get_connections(self, S):
        solution = set()
        while S:
            # Count element frequency across remaining subsets
            freq = Counter()
            for s in S:
                freq.update(s)
    
            # Pick element that hits maximum subsets
            best_element = max(freq, key=freq.get)
            solution.add(best_element)
    
            # Remove all subsets hit by best_element
            S = [s for s in S if best_element not in s]
        return list(solution)        
        
    
    def create_adjancency_matrix(self, C):
        A = np.zeros((len(C), len(C)), dtype = bool);
        for i, C_i in enumerate(C):
            S =  C_i.copy();
            idxs = np.arange(len(S));
            
            #remove self-set and its index
            S.pop(i);
            idxs = idxs[idxs != i];
            
            #remove empty sets and their indices
            idxs_empty = [i for i, s in zip(idxs, S) if not s];
            S = [s for s in S if s];
            A[i, idxs_empty] = True;
            
            #get optimal subset of connections
            optimal_idxs = self.get_connections(S);
            try:
                A[i, optimal_idxs] = True;
            except:
                print(optimal_idxs)
        return A;
    
    """Graph construction"""
    def construct_graph(self, X):
        n, d = X.shape;
        print("Creating distance matrix...");
        D = self.create_distance_matrix(X);
        print("Creating candidate matrix...");
        C = self.create_candidate_matrix(D);
        print("Creating adjecancy matrix...");
        self.A = self.create_adjancency_matrix(C);
        return;
    
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
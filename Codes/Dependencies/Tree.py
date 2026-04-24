import numpy as np
from sklearn.cluster import KMeans

class Tree():
    def __init__(self):
        self.leaf_size = 1;
        self.X = None;
        self.root = None;
        
    def generate_tree(self, indices, center, depth = 0):
        if(len(indices) == 1): 
            return {"indices": indices,
                    "center": center,
                    "left": None,
                    "right": None
                };
        elif(len(indices) == 2):
            idx_C1 = indices[0];
            idx_C2 = indices[1];
            center_C1 = self.X[idx_C1];
            center_C2 = self.X[idx_C2];
            return {"center" : center,
                    "indices" : None,
                    "left" : self.generate_tree(np.array([idx_C1]), center_C1, depth+1),
                    "right" : self.generate_tree(np.array([idx_C2]), center_C2, depth+1)
                };
        else:
            kmeans = KMeans(n_clusters = 2, init = "random").fit(self.X[indices]);
            idx_C1 = indices[np.where(kmeans.labels_ == 0)[0]];
            idx_C2 = indices[np.where(kmeans.labels_ == 1)[0]];
            
            if(len(idx_C1) == len(indices)):
                print("one cluster");
            
            center_C1 = np.sum(self.X[idx_C1], axis = 0) / len(self.X[idx_C1]);
            center_C2 = np.sum(self.X[idx_C2], axis = 0) / len(self.X[idx_C2]);
            return {"center" : center,
                    "indices" : None,
                    "left" : self.generate_tree(idx_C1, center_C1, depth+1),
                    "right" : self.generate_tree(idx_C2, center_C2, depth+1)
                };
        
    def fit(self, X):
        n, m = X.shape;
        self.X = X;
        center = np.sum(self.X, axis = 0) / len(self.X);
        self.root = self.generate_tree(np.arange(n), center);
    
    def find_nearest_neighbor(self, test_point, root):
        if(root["left"] == None and root["right"] == None):
            return root["indices"], 0;
        else:
            distance_left = np.linalg.norm(test_point - root["left"]["center"]);
            distance_right = np.linalg.norm(test_point - root["right"]["center"]);
            if(distance_left <= distance_right):
                neighbors, n_distances = self.find_nearest_neighbor(test_point, root["left"]);
                return neighbors, n_distances + 1;
            else:
                neighbors, n_distances = self.find_nearest_neighbor(test_point, root["right"]);
                return neighbors, n_distances + 1;

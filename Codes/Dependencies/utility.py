import math 
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import csv
import math

def euclidean_distance(p1, p2):
    """
    Computes Euclidean distance between two points.
    p1, p2: iterables (list/tuple) of coordinates
    """
    if len(p1) != len(p2):
        raise ValueError("Points must have the same dimension")

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def minkowski_distance(x, y, p=2):
    """
    Compute Minkowski distance between two vectors.
    
    Parameters:
    x, y : lists or tuples of equal length
    p    : order of the norm (p >= 1)
    
    Returns:
    float
    """
    if len(x) != len(y):
        raise ValueError("Vectors must have the same length")

    return sum(abs(a - b) ** p for a, b in zip(x, y)) ** (1 / p)

def cosine_distance(x, y):
    """
    Cosine distance = 1 - cosine similarity
    """
    dot = sum(a * b for a, b in zip(x, y))
    norm_x = math.sqrt(sum(a * a for a in x))
    norm_y = math.sqrt(sum(b * b for b in y))

    if norm_x == 0 or norm_y == 0:
        raise ValueError("Zero vector encountered")

    cosine_similarity = dot / (norm_x * norm_y)
    return 1 - cosine_similarity


def generate_random_uniform_dataset(n, d):
    return np.random.uniform(0, 1, (n, d));

def generate_clustered_dataset(n, d, c):
    X, y = make_blobs(
        n_samples=n,        # total points
        centers=c,            # number of clusters
        n_features=d,         # 2D data
        cluster_std=2.0,      # spread of clusters
        random_state=42
        );
    return X;

def generate_moons_dataset(n, d = 2):
    X, y = make_moons(
        n_samples=n,
        noise=0.05,
        random_state=42
    )
    return X;

def load_SIFT(path, file_name):
    # Read entire file as int32
    data = np.fromfile(path + file_name, dtype=np.int32)
    # First int32 is dimension
    dim = data[0]  
    # Reshape
    data = data.reshape(-1, dim + 1)  
    # Remove dimension column
    vectors = data[:, 1:].view(np.float32) 
    return vectors

def load_dataset(path, file_name):
    data = pd.read_csv(path + file_name)    
    X = data.iloc[:, :-1].to_numpy(dtype=np.float64);
    scaler = MinMaxScaler();
    return scaler.fit_transform(np.unique(X, axis=0));

def save_to_csv(data, filename, header):
    file_exists = os.path.isfile(".//..//Results//Results//" + filename + ".csv")
    with open(".//..//Results//Results//" + filename + ".csv", mode="a", newline="") as f:
        writer = csv.writer(f);
        # write header only once
        if not file_exists:
            writer.writerow(header)
        # write data row
        writer.writerow(data.tolist());
        
def find_recall(n_bf,n_mjg):
    count = 0;
    total = 0;
    n = len(n_bf);
    for i in range(n):
        count += len(set(n_mjg[i]) - set(n_bf[i]));
        total += len(n_bf[i]);
    recall = (total - count)/total;
    return np.round(recall, 2);

def get_efficiency(pe_bf, pe_mjg):
    total_bf = 0;
    total_mjg = 0;
    n = len(pe_bf);
    for i in range(n):
        total_bf += pe_bf[i];
        total_mjg += pe_mjg[i];
    return np.round((total_bf - total_mjg)/total_bf, 2);

def get_total(A):
    total = 0;
    for e in A:
        total += e;
    return np.round(total, 2);
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import time
from multiprocessing import freeze_support
from sklearn.model_selection import train_test_split
from Dependencies.utility import generate_random_uniform_dataset, generate_moons_dataset, generate_clustered_dataset
from Dependencies.BF import BF
from Dependencies.utility import save_to_csv, find_recall, get_efficiency, get_total, load_SIFT, load_dataset
from MJG.AMJG_PE import AMJG_PE
from sklearn.datasets import fetch_openml

def brute_force_knn(X, query, n_neighbors=5):
    # Compute squared Euclidean distances
    dists = np.sum((X - query) ** 2, axis=1)

    # Get indices of k smallest distances
    idx = np.argpartition(dists, n_neighbors)[:n_neighbors]

    # Sort them properly
    idx = idx[np.argsort(dists[idx])]

    return idx

def print_attributes(G):
    edges = G.find_total_edges();
    avg_Degree, max_degree  = G.find_degree();
    print("Total Edges = ", edges);
    print("Average Degree = ", np.round(avg_Degree, 2));
    print("Max Degree = ", max_degree);
    return edges, np.round(avg_Degree, 2), max_degree;
    
def test_reachability(G, X, n):
    reachability = np.zeros((n, n), dtype = bool);
    for i in range(n):
        for j in range(n):
            if(i == j):
                reachability[i][j] = True;
            else:
                reachability[i][j] = G.test_reachability(X, i, j);
    print("Number of unreachable paths = ", (n*n) - np.sum(reachability));
    
def generate_dataset(d_type, n, d, c = 3, path = ""):
    if(d_type == "-r"):
        X = generate_random_uniform_dataset(n, d);
        file_name = f"RU_{n}x{d}";
        X_train, X_test = train_test_split(X, test_size=100, random_state=42, shuffle=False);
    elif(d_type == "-c"):
        X = generate_clustered_dataset(n, d, c);
        file_name = f"C_{n}x{d}x{c}";
        X_train, X_test = train_test_split(X, test_size=100, random_state=42, shuffle=False);
    elif(d_type == "-m"):
        X = generate_moons_dataset(n);
        file_name = f"M_{n}x2";
        X_train, X_test = train_test_split(X, test_size=100, random_state=42, shuffle=False);
    elif(d_type == "-mnist"):
        X = fetch_openml("mnist_784", version = 1, as_frame= False);
        file_name = "MNIST";
        X = X["data"].astype(np.float32)/255.0;
        X_train, X_test = train_test_split(X, test_size=100, random_state=42, shuffle=False);
    elif(d_type == "-sift"):
        X_train = load_SIFT(".//..//Datasets//sift//", "sift_learn.fvecs");
        file_name = "SIFT";
        X_test = load_SIFT(".//..//Datasets//sift//", "sift_query.fvecs");
        if X_test.shape[0] > 10:
            X_test = X_test[:10, :]
    elif(d_type == "-siftsmall"):
        X_train = load_SIFT(".//..//Datasets//siftsmall//", "sift_learn.fvecs");
        file_name = "SIFT_SMALL";
        X_test = load_SIFT(".//..//Datasets//siftsmall//", "sift_query.fvecs");
    elif(d_type == "-shuttle"):
        path = ".\\..\\Datasets\\";
        file_name = "SHUTTLE";
        X = load_dataset(path, file_name + ".csv");
        X_train, X_test = train_test_split(X, test_size=100, random_state=42, shuffle=True);
    else:
        print("Unrecoginized type of dataset");
        print("Valid Types are:");
        print("-r : Random Uniform");
        print("-c : Clustered (Default three clusters)");
        print("-m : Shape based moons (fixed two dimensions)");
        print("-mnist : Standard Deep Learning MNIST dataset");
        sys.exit(-1);
    
    return X_train, X_test, file_name;
        
        
def main():
    n = 50100;
    d = 100;
    k = 10
    d_type = "-sift";
    #additional = 30;
    X_train, X_test, file_name = generate_dataset(d_type, n, d);
    amjg = AMJG_PE();
    st_const = time.time();
    amjg.construct_graph(X_train);
    end_const = time.time() - st_const;
    print("-------------AMJ-Graph--------------");
    e, ad, md = print_attributes(amjg);
    print("Construction Time = ", round(end_const, 2));
    
    """Search Neighbors Brute Force"""
    BF_results = {"neighbors": list()};
    for query_point in X_test:
        neighbors = brute_force_knn(X_train, query_point, n_neighbors = k);
        BF_results["neighbors"].append(neighbors);
    
    for additional in range(0, 100):
        """Search Neighbors MJ-Graph"""
        MJ_G_results = {"neighbors": list(), "search time" : list(), "points explored": list()};
        for i in range(len(X_test)):
            s_time = time.time();
            neighbors, search_explored, reach_explored = amjg.query(X_test[i], X_train, n_neighbors = k, n_additional = additional);
            e_time = time.time();
            MJ_G_results["neighbors"].append(neighbors);
            MJ_G_results["search time"].append(e_time - s_time);
            MJ_G_results["points explored"].append(search_explored);
        
        recall = find_recall(BF_results["neighbors"], MJ_G_results["neighbors"]);
        st_mj = get_total(MJ_G_results["search time"])
        pe_mj = get_total(MJ_G_results["points explored"]);
        
        header = ["Dataset", f"Recall@{k}", "DC", "Avg. DC", "ST", "Lambda", "DM"];
        data = np.array([file_name, str(recall), str(pe_mj),str(round(pe_mj / X_test.shape[0], 2)), str(st_mj), str(additional), "l2"]);
        save_to_csv(data, "RDC_Others", header);

if __name__ == "__main__":
    freeze_support();   # REQUIRED on Windows
    main();





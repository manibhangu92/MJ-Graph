import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import time
from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support

from Dependencies.BF import BF
from Dependencies.utility import save_to_csv, find_recall, get_efficiency, get_total
from Dependencies.utility import load_dataset
from MJG.AMJG_PE import AMJG_PE


def print_attributes(G):
    edges = G.find_total_edges();
    avg_Degree, max_degree  = G.find_degree();
    print("Total Edges = ", edges);
    print("Average Degree = ", np.round(avg_Degree, 2));
    print("Max Degree = ", max_degree);
    return edges, np.round(avg_Degree, 2), max_degree;
   
def main():
    path = ".\\..\\Datasets\\";
    file_name = "SHUTTLE"
    k = 10
    additional = 20;
    X = load_dataset(path, file_name + ".csv");
    X_train, X_test = train_test_split(X, test_size=0.30, random_state=42, shuffle=False);
    amjg = AMJG_PE();
    amjg.construct_graph(X_train);
    print("-------------AMJ-Graph--------------");
    e, ad, md = print_attributes(amjg);
    
    
    """Search Neighbors MJ-Graph"""
    MJ_G_results = {"neighbors": list(), "search time" : list(), "points explored": list()};
    for i in range(len(X_test)):
        s_time = time.time();
        neighbors, search_explored, reach_explored = amjg.query(X_test[i], X_train, n_neighbors = k, n_additional = additional);
        e_time = time.time();
        MJ_G_results["neighbors"].append(neighbors);
        MJ_G_results["search time"].append(e_time - s_time);
        MJ_G_results["points explored"].append(search_explored);

    """Search Neighbors Brute Force"""
    bf = BF();
    BF_results = {"neighbors": list(), "search time" : list(), "points explored": list()};
    for query_point in X_test:
        s_time = time.time();
        neighbors = bf.query(query_point, X_train, n_neighbors = k);
        e_time = time.time();
        BF_results["neighbors"].append(neighbors);
        BF_results["search time"].append(e_time - s_time);
        BF_results["points explored"].append(X_train.shape[0]);
    
    recall = find_recall(BF_results["neighbors"], MJ_G_results["neighbors"]);
    eff = get_efficiency(BF_results["points explored"], MJ_G_results["points explored"]);
    st_bf = get_total(BF_results["search time"]);
    st_mj = get_total(MJ_G_results["search time"])
    pe_bf = get_total(BF_results["points explored"]);
    pe_mj = get_total(MJ_G_results["points explored"]);
    print(f"Recall@{k} : ", recall);
    print("Efficiency = ",eff);
    print("Search Time BF : ", st_bf);
    print("Search Time MJ-Graph : ", st_mj);
    print("Distance Computations by BF : ", pe_bf);
    print("Distance Computations by MJ-Graph : ", pe_mj);
    print("Avg. Distance Computations by MJ-Graph : ", round(pe_mj/X_test.shape[0], 2));
    
    header = ["Dataset", "Edges", "Avg. Degree", "Max. Degree", f"Recall@{k}", "DC", "Avg. DC", "Efficiency", "ST", "Lambda" , "DC_BF", "ST_BF", "DM"];
    data = np.array([file_name, str(e), str(ad), str(md), str(recall), str(pe_mj),str(round(pe_mj / X_test.shape[0], 2)), str(eff), str(st_mj), str(additional), str(pe_bf), str(st_bf), str("l2")]);
    save_to_csv(data, "MJG", header);

if __name__ == "__main__":
    freeze_support();   # REQUIRED on Windows
    main();




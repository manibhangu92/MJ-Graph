import numpy as np
import time
import csv
from MJG.AMJG_E import AMJG_E
from MJG.GMJG import GMJG

def generate_random_uniform_dataset(n, d):
    return np.random.uniform(0, 1, (n, d));
    
def construct_AMJG(X):
    start = time.time();
    G = AMJG_E();
    G.construct_graph(X);
    end = time.time() - start;
    return G, end;

def construct_GMJG(X):
    start = time.time();
    G = GMJG();
    G.construct_graph(X);
    end = time.time() - start;
    return G, end;

def get_attributes(G):
    avg_Degree, max_degree  = G.find_degree();
    return G.find_total_edges(), avg_Degree, max_degree;
     
    
    
def test_reachability(G, X, n):
    reachability = np.zeros((n, n), dtype = bool);
    for i in range(n):
        for j in range(n):
            if(i == j):
                reachability[i][j] = True;
            else:
                reachability[i][j] = G.test_reachability(X, i, j);
    print("Number of unreachable paths = ", (n*n) - np.sum(reachability));
    
    
def save_data(header, data):
    for filename, inner_dict in data.items():
        csv_filename = f".//..//Results//Sensitivity//{filename}.csv"
    
        with open(csv_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(header)
            
            # Write rows
            for dim, values in inner_dict.items():
                if len(values) != 10:
                    raise ValueError(f"{dim} in {filename} does not have 10 elements")
                writer.writerow([dim] + values)

    print(f"Saved: {csv_filename}")
    
n = 1000;
dim = [i for i in range(10, 101, 10)];

results = {
            "gmjg":{
                    "avg_degree": [],
                    "max_degree" : [],
                    "edges" : [],
                    "ct" : []
                },
           "amjg":{
                   "avg_degree": [],
                   "max_degree" : [],
                   "edges" : [],
                   "ct" : []
               }
           }
for d in dim:
    print("iteration = ", d/10);
    X = generate_random_uniform_dataset(n, d);
    print("-------------AMJ-Graph--------------");
    amjg, amjg_ct = construct_AMJG(X);
    edges, avg_degree, max_degree = get_attributes(amjg);
    results["amjg"]["edges"].append(edges);
    results["amjg"]["max_degree"].append(max_degree);
    results["amjg"]["avg_degree"].append(np.round(avg_degree, 2));
    results["amjg"]["ct"].append(np.round(amjg_ct, 2));
    print("Construction Time = ", np.round(amjg_ct, 2));
    
    print("------------GMJ-Graph--------------");
    mjg, mjg_ct = construct_GMJG(X);
    edges, avg_degree, max_degree = get_attributes(mjg);
    results["gmjg"]["edges"].append(edges);
    results["gmjg"]["max_degree"].append(max_degree);
    results["gmjg"]["avg_degree"].append(np.round(avg_degree, 2));
    results["gmjg"]["ct"].append(np.round(mjg_ct, 2));
    print("Construction Time = ", np.round(mjg_ct, 2));
    
save_data(["Dim"] + dim, results);
    



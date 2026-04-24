import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import networkx as nx
import matplotlib.pyplot as plt
from MJG.AMJG_E import AMJG_E
from MJG.GMJG import GMJG

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
    
def plot_graph(A, X, title, graph_name):
    n, d = X.shape;
    G = nx.Graph();
    #add nodes
    for i in range(n):
        G.add_node(i, pos=(X[i,0], X[i,1]));
    #add edges
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j]:
                G.add_edge(i, j);
    pos = nx.get_node_attributes(G, 'pos')            
    plt.figure()
    nx.draw(G, pos, with_labels=False, node_size=50);
    plt.title(title, fontname="Times New Roman", fontweight="bold", fontsize=22);
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(".//..//Results//Graphs//" + graph_name+".png", dpi=3000, bbox_inches="tight"
)
    plt.show();
    
def plot_dataset(X, title, dataset_name):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.title( title, fontname="Times New Roman", fontweight="bold", fontsize=22);
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(".//..//Results//Graphs//" + dataset_name+".png", dpi=3000, bbox_inches="tight");
    plt.show()

def plot_AMJG(X, dataset_name):
    G = AMJG_E();
    G.construct_graph(X);
    plot_graph(G.A, X, "Approximate MJ-Graph", "AMJG_" +dataset_name);
    return G;

def plot_MJG(X, dataset_name):
    G = GMJG();
    G.construct_graph(X);
    plot_graph(G.A, X, "GMJ-Graph", "GMJG_" +dataset_name);
    return G;

def print_attributes(G):
    print("Total Edges = ", G.find_total_edges());
    avg_Degree, max_degree  = G.find_degree(); 
    print("Average Degree = ", avg_Degree);
    print("Max Degree = ", max_degree);
    
def test_reachability(G, X, n):
    reachability = np.zeros((n, n), dtype = bool);
    for i in range(n):
        for j in range(n):
            if(i == j):
                reachability[i][j] = True;
            else:
                reachability[i][j] = G.test_reachability(X, i, j);
    print("Number of unreachable paths = ", (n*n) - np.sum(reachability));
    
n = 50;
d = 2;
c = 3;

dataset = "-m";
if(dataset == "-r"):
    X = generate_random_uniform_dataset(n, d);
    plot_dataset(X, "Random Uniform Dataset", "Random Uniform");
    mjg = plot_MJG(X, "Random Uniform");
    amjg = plot_AMJG(X, "Random Uniform");
elif(dataset == "-c"):
    X = generate_clustered_dataset(n, d, c);
    plot_dataset(X, "Clustered Dataset", "Clustered");
    mjg = plot_MJG(X, "Clustered");
    amjg = plot_AMJG(X, "Clustered");
elif(dataset == "-m"):
    X = generate_moons_dataset(n, d);
    plot_dataset(X, "Shape-based Clustered Dataset", "Moons");
    mjg = plot_MJG(X, "Moons");
    amjg = plot_AMJG(X, "Moons");
else:
    print("dataset not recoginized");
    
print("------------GMJ-Graph--------------");
print_attributes(mjg);
test_reachability(mjg, X, X.shape[0]);
print("-------------AMJ-Graph--------------");
print_attributes(amjg);
test_reachability(amjg, X, X.shape[0]);



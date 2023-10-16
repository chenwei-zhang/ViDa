import numpy as np
import networkx as nx
import pickle
import copy
from sklearn.preprocessing import MinMaxScaler
import argparse


# Build the edges
def get_all_edges(coord_id_S, trj_id):
    all_nodes = coord_id_S
    all_edges_temp = []

    for previous, current in zip(all_nodes, all_nodes[1:]):
        all_edges_temp.append((previous, current))

    indices_to_delete = trj_id[:-1]
    # Sort the indices in reverse order so that deleting elements won't affect subsequent indices
    indices_to_delete = sorted(indices_to_delete, reverse=True)

    all_edges = copy.deepcopy(all_edges_temp)
    for index in indices_to_delete:
        del all_edges[index]

    return all_edges


# construct the modified weighted undirected graph   
def build_MUG(all_edges, SIMS_HT_uniq):
    MUG = nx.Graph()
    
    for i in range(len(all_edges)):
        idx0 = all_edges[i][0]
        idx1 = all_edges[i][1]
        
        if SIMS_HT_uniq[idx0] < SIMS_HT_uniq[idx1]:
            weight = SIMS_HT_uniq[idx0]
        else:
            weight = SIMS_HT_uniq[idx1]
            
        if SIMS_HT_uniq[idx0] == 0 or SIMS_HT_uniq[idx1] == 0:
            weight = SIMS_HT_uniq[idx0] + SIMS_HT_uniq[idx1]
            
        MUG.add_edge(int(all_edges[i][0]), int(all_edges[i][1]), weight=float(weight))
    
    return MUG


# calculate MPT for each node 
def collect_and_pad_shortest_paths(MUG, SIMS_HT_uniq, knn=100):
    X_j = []
    D_ij = []

    for i in range(len(SIMS_HT_uniq)):
    
        length = nx.single_source_dijkstra_path_length(MUG, i)
        length_arr = np.array(list(length.items()), dtype=object)
        
        # get the nearest knn nodes
        X_j.append(length_arr[1:knn+1, 0])
        D_ij.append(length_arr[1:knn+1, 1])
        
        if i % 5000 == 0:
            print(f'finished {i} nodes')

    # padd the X_j and D_ij to the same length
    for i in range(len(X_j)):
        if len(X_j[i]) < knn:
            X_j[i] = np.pad(X_j[i], (0, knn-len(X_j[i])), 'constant', constant_values=i)
            D_ij[i] = np.pad(D_ij[i], (0, knn-len(D_ij[i])), 'constant', constant_values=0)

    scaler = MinMaxScaler(feature_range=(0,3)) 
    norm_Dij = scaler.fit_transform(D_ij)
    
    return np.array(X_j, dtype=int), np.array(norm_Dij, dtype=float)


# calculate the probability of being visited during a simulated trajectory 
def calculate_P_tot(coord_id_S, trj_id, SIMS_HT_uniq):
    split_id = trj_id + 1  # index for split to each trajectory
    P_tot = np.zeros(len(SIMS_HT_uniq))

    for i in range(len(split_id)):
        if i == 0:
            trj = set(coord_id_S[0:split_id[i]])
        else:
            trj = set(coord_id_S[split_id[i-1]:split_id[i]])

        P_tot[list(trj)] += 1

    P_tot = P_tot / 100

    return P_tot


# calculate the graph edit distance between two graphs
def edit_distance(adj1, adj2):
    # Calculate edit distance based on its adjacency matrix
    edit_dist = np.sum(np.abs(adj1-adj2),dtype=int)
    
    return edit_dist


# Calculate the ged between X_i and X_j
def collect_ged(X_j, SIMS_adj_uniq):
    ED_ij = []

    for i in range(X_j.shape[0]):
        ed_ij = []
        for j in range(X_j.shape[1]):
            ed_ij.append(edit_distance(SIMS_adj_uniq[i], SIMS_adj_uniq[X_j[i,j]]))
        ED_ij.append(ed_ij)
    ED_ij = np.array(ED_ij, dtype=int)

    return ED_ij



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', required=True, help='preprocessed data, time data')
    parser.add_argument('--holdtime', required=True, help='average holding time for each node')
    parser.add_argument('--adjmat', required=True, help='ajacency matrix for each node')
    parser.add_argument('--outpath', required=True, help='output minimum passage time distance')

    args = parser.parse_args()

    inpath = args.inpath
    holdtime = args.holdtime
    outpath = args.outpath
    adjmat = args.adjmat
    
    # Load the data
    print(f"[compdist] Loading preprocessed index from {inpath}")
    
    with open(inpath, 'rb') as file:
        loaded_data = pickle.load(file)
        
    coord_id_S = loaded_data["coord_id_S"]
        
    
    print(f"[compdist] Loading average holding time from {holdtime}")

    with open(holdtime, 'rb') as file:
        loaded_data = pickle.load(file)
        
    SIMS_HT_uniq = loaded_data["SIMS_HT_uniq"]
    trj_id = loaded_data["trj_id"]
    
    
    print(f"[compdist] Loading adjacency matrix from {adjmat}")
    
    with open(adjmat, 'rb') as file:
        loaded_data = pickle.load(file)
        
    SIMS_adj_uniq = loaded_data["SIMS_adj_uniq"]


    # Build the edges
    print("[compdist] Building the edges")
    
    all_edges = get_all_edges(coord_id_S, trj_id)
    
    # Construct the modified weighted undirected graph
    print("[compdist] Constructing the modified weighted undirected graph")
    
    MUG = build_MUG(all_edges, SIMS_HT_uniq)
    
    # Collect the shortest path for each node
    print("[compdist] Collecting the MPT and its corresponding index")
    
    X_j, D_ij = collect_and_pad_shortest_paths(MUG, SIMS_HT_uniq)
    
    # Calculate the probability of being visited during a simulated trajectory
    print("[compdist] Calculating the node probability")
    
    P_tot = calculate_P_tot(coord_id_S, trj_id, SIMS_HT_uniq)
    
    
    # Calculate the graph edit distance between X_i and X_j
    print("[compdist] Calculating the graph edit distance")
    
    ED_ij = collect_ged(X_j, SIMS_adj_uniq)
    
    
    # save npz file for shortest path
    print(f"[compdist] Saving MPT and GED to {outpath}")
    
    with open(outpath, 'wb') as f:
        np.savez(f,
                 X_j = X_j,
                 D_ij = D_ij,
                 ED_ij = ED_ij,
                 P_tot = P_tot
             )
        
    print("[compdist] Done!")
    
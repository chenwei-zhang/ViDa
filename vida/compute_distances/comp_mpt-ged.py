import numpy as np
import networkx as nx
import copy
from sklearn.preprocessing import MinMaxScaler
import argparse
import time
import heapq

# Build the edges
def get_all_edges(indices_all, trj_id):
    all_nodes = indices_all
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

'''
# construct the modified weighted undirected graph   
def build_mug(all_edges, hold_time_uniq):
    MUG = nx.Graph()
    
    for i in range(len(all_edges)):
        idx0 = all_edges[i][0]
        idx1 = all_edges[i][1]
        
        if hold_time_uniq[idx0] < hold_time_uniq[idx1]:
            weight = hold_time_uniq[idx0]
        else:
            weight = hold_time_uniq[idx1]
            
        if hold_time_uniq[idx0] == 0 or hold_time_uniq[idx1] == 0:
            weight = hold_time_uniq[idx0] + hold_time_uniq[idx1]
            
        MUG.add_edge(int(all_edges[i][0]), int(all_edges[i][1]), weight=float(weight))
    
    return MUG

# calculate MPT for each node 
def collect_and_pad_shortest_paths(MUG, knn=100):
    x_j = []
    d_ij = []

    for i in range(len(MUG.nodes)):
    
        length = nx.single_source_dijkstra_path_length(MUG, i)
        length_arr = np.array(list(length.items()), dtype=object)
        
        # get the nearest knn nodes
        x_j.append(length_arr[1:knn+1, 0])
        d_ij.append(length_arr[1:knn+1, 1])
        
        if i % 5000 == 0:
            print(f'finished {i} nodes')

    # padd the x_j and d_ij to the same length
    for i in range(len(x_j)):
        if len(x_j[i]) < knn:
            x_j[i] = np.pad(x_j[i], (0, knn-len(x_j[i])), 'constant', constant_values=i)
            d_ij[i] = np.pad(d_ij[i], (0, knn-len(d_ij[i])), 'constant', constant_values=0)

    scaler = MinMaxScaler(feature_range=(0,3)) 
    norm_dij = scaler.fit_transform(d_ij)
    
    return np.array(x_j, dtype=int), np.array(norm_dij, dtype=float)
'''


# construct weighted directed graph
def build_wdg(all_edges, hold_time_uniq):
    DG = nx.DiGraph()
    for i in range(len(all_edges)):
        weight = hold_time_uniq[all_edges[i][0]]
        DG.add_edge(int(all_edges[i][0]), int(all_edges[i][1]), weight=float(weight))
    
    return DG  


def dijkstra_n_shortest_paths(graph, source, n_neigh=100):
    # Initialize data structures
    visited = set()
    distances = {node: float('infinity') for node in graph}
    distances[source] = 0
    priority_queue = [(0, source)]
    paths = []

    # Main loop
    while priority_queue and len(paths) < n_neigh:
                
        _, current_node = heapq.heappop(priority_queue)
        
        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor, weight in graph[current_node].items():
            if neighbor not in visited:
                tentative_distance = distances[current_node] + weight['weight']

                if tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    heapq.heappush(priority_queue, (tentative_distance, neighbor))
                            
        paths.append((source, current_node, distances[current_node]))
        
    return np.array(paths)


def calculate_mpt(G,n_neigh=100):
    x_j, d_ij = [], []
    
    for  i in range(len(G.nodes)):
        # calculate the shortest path from node i to its 100 nearest neighbors including itself
        shortest_path_i = dijkstra_n_shortest_paths(G, i)
        x_j.append(shortest_path_i[:,1].astype(int))
        d_ij.append(shortest_path_i[:,2].astype(float))
        
        # pad the x_j and d_ij to the same length
        if len(x_j[i]) < n_neigh:
            x_j[i] = np.pad(x_j[i], (0, n_neigh-len(x_j[i])), 'constant', constant_values=i)
            d_ij[i] = np.pad(d_ij[i], (0, n_neigh-len(d_ij[i])), 'constant', constant_values=0)
    
    # normalize the d_ij
    min_val = np.min(d_ij)
    max_val = np.max(d_ij)
    norm_dij = (d_ij - min_val) / (max_val - min_val) 
    
    return np.array(x_j, dtype=int), norm_dij


# calculate the probability of being visited during a simulated trajectory 
def calculate_prob(indices_all, trj_id, hold_time_uniq):
    split_id = trj_id + 1  # index for split to each trajectory
    p_i = np.zeros(len(hold_time_uniq))

    for i in range(len(split_id)):
        if i == 0:
            trj = set(indices_all[0:split_id[i]])
        else:
            trj = set(indices_all[split_id[i-1]:split_id[i]])

        p_i[list(trj)] += 1

    p_i = p_i / 100

    return p_i


# calculate the graph edit distance between two graphs
def edit_distance(adj1, adj2):
    # Calculate edit distance based on its adjacency matrix
    edit_dist = np.sum(np.abs(adj1-adj2),dtype=int)
    
    return edit_dist


# Calculate the ged between X_i and x_j
def calculate_ged(x_j, adj_uniq):
    e_ij = []

    for i in range(x_j.shape[0]):
        ed_ij = []
        for j in range(x_j.shape[1]):
            ed_ij.append(edit_distance(adj_uniq[i], adj_uniq[x_j[i,j]]))
        e_ij.append(ed_ij)
    e_ij = np.array(e_ij, dtype=int)

    return e_ij



if __name__ == '__main__':
    # Record the start time
    start_time = time.time()

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
    print(f"[Comp_dist] Loading preprocessed index from {inpath}")
    
    loaded_data = np.load(inpath)
    
    indices_all = loaded_data["indices_all"]
        
    
    print(f"[Comp_dist] Loading average holding time from {holdtime}")

    loaded_data = np.load(holdtime)
        
    hold_time_uniq = loaded_data["hold_time_uniq"]
    trj_id = loaded_data["trj_id"]
    
    
    print(f"[Comp_dist] Loading adjacency matrix from {adjmat}")

    loaded_data = np.load(adjmat)
    
    adj_uniq = loaded_data["adj_uniq"]


    # Build the edges
    print("[Comp_dist] Building the edges")
    
    all_edges = get_all_edges(indices_all, trj_id)
    
    # Construct the modified weighted undirected graph
    print("[Comp_dist] Constructing the modified weighted undirected graph")
    
    DG = build_wdg(all_edges, hold_time_uniq)
    
    
    # Collect the shortest path for each node
    print("[Comp_dist] Collecting the MPT and its corresponding index")
    
    x_j, d_ij = calculate_mpt(DG)
    
    
    # Calculate the probability of being visited during a simulated trajectory
    print("[Comp_dist] Calculating the node probability")
    
    p_i = calculate_prob(indices_all, trj_id, hold_time_uniq)
    
    
    # Calculate the graph edit distance between X_i and x_j
    print("[Comp_dist] Calculating the graph edit distance")
    
    e_ij = calculate_ged(x_j, adj_uniq)
    
    
    # save pickle file for shortest path
    print(f"[Comp_dist] Saving MPT and GED to {outpath}")
    
    data_to_save = {
    "x_j": x_j,
    "d_ij": d_ij,
    "e_ij": e_ij,
    "p_i": p_i,
    }
    
    np.savez_compressed(outpath, **data_to_save)
        
    print("[Comp_dist] Done!")
    
    # Record the end time
    end_time = time.time()
    
    # Print the time elapsed
    print(f"[Comp_dist] Elapsed Time: {(end_time - start_time):.3f} seconds")
    
    
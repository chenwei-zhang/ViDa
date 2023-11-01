import numpy as np
import networkx as nx
import copy
import heapq
from annoy import AnnoyIndex


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
    x_dj, d_ij = [], []
    
    for  i in range(len(G.nodes)):
        # calculate the shortest path from node i to its 100 nearest neighbors including itself
        shortest_path_i = dijkstra_n_shortest_paths(G, i)
        x_dj.append(shortest_path_i[:,1].astype(int))
        d_ij.append(shortest_path_i[:,2].astype(float))
        
        # pad the x_dj and d_ij to the same length
        if len(x_dj[i]) < n_neigh:
            x_dj[i] = np.pad(x_dj[i], (0, n_neigh-len(x_dj[i])), 'constant', constant_values=i)
            d_ij[i] = np.pad(d_ij[i], (0, n_neigh-len(d_ij[i])), 'constant', constant_values=0)
    
    # normalize the d_ij
    min_val = np.min(d_ij)
    max_val = np.max(d_ij)
    norm_dij = (d_ij - min_val) / (max_val - min_val) 
    
    return np.array(x_dj, dtype=int), norm_dij


def calculate_ged(adj_uniq,n_neigh=100):
    x_ej, e_ij = [], []

    num_graphs = len(adj_uniq)
    num_features = len(adj_uniq[0]) * len(adj_uniq[0][0])

    # Initialize AnnoyIndex
    annoy_index = AnnoyIndex(num_features, 'manhattan')

    # Add vectors to the index
    for i, matrix in enumerate(adj_uniq):
        vector = matrix.flatten()
        annoy_index.add_item(i, vector)

    # Build the index
    annoy_index.build(10)  # may need to tune this parameter

    for i in range(num_graphs):
        indices, distances = annoy_index.get_nns_by_item(i, n_neigh, include_distances=True)
        e_ij.append(distances)
        x_ej.append(indices)
    
    return np.array(x_ej), np.array(e_ij)


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



    
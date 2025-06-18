import numpy as np
import networkx as nx
from annoy import AnnoyIndex
import tqdm
import nupack as nu 

def get_transitions(indices_all, endpoints):
    
    '''
    Returns the sampled elementary transitions 
    '''

    edges = [(indices_all[i], indices_all[i+1]) for i in range(len(indices_all)-1) if i not in endpoints[:-1]]

    return np.unique(edges, axis=0)


def build_wdg(transitions, hold_time_uniq):

    '''
    Returns a graph for the sampled Multistrand states and transitions
        Nodes: sampled secondary structures  
        Edges: sampled elementary transitions
        Weights: weight of (i -> j) is the empirical holding time of state i (independent of j)   
    '''

    # TODO: Instead of the empirical holding time for the weights, consider: 
    #  (a) empirical transition time
    #  (b) expected holding time, or
    #  (c) expected transition time  

    # TODO: Use enumerated transitions rather than sampled transitions
 
    G = nx.DiGraph()
  
    # Add nodes
    state_ids = range(len(hold_time_uniq))
    G.add_nodes_from(state_ids)

    # Add edges
    weighed_edges = [(transitions[i][0], transitions[i][1], hold_time_uniq[transitions[i][0]]) for i in range(len(transitions))]
    G.add_weighted_edges_from(weighed_edges)
   
    return G  


def calculate_mpt(G, k=100):

    '''
    Uses the digraph representation of the sampled Multistrand states/transitions 
    to calculate a time-based distance from each state to its nearest k states
 
    dist(x0,xn) = min_{all paths x0->x1...->xn in G} sum_{i=0}^{n-1} edge-weight(xi->x{i+1}) 
                = min_{all paths x0->x1...->xn in G} sum_{i=0}^{n-1} empirical-holding-time(xi)

    Returns: 
        nearest_neighbours[i]: 
            indices of the k states nearest to state i (ties broken arbitrarily)
        nearest_distances[i][k]: 
            the distance between state i and state nearest_distances[i][k], 
    '''

    n_states = len(G.nodes)

    k = min(k, n_states) # TODO: add warning for k > n_states
    
    # initalize each node's k nearest neighbours as itself, with distance 0.0
    nearest_distances = np.zeros((n_states, k), dtype=float)
    nearest_neighbours = np.tile(np.arange(n_states),(k,1)).T 

    # TODO: consider using NetworKit implementation of all-pairs-shortest-path 
    # TODO: tune cutoff based on graph size/properties 
    results = nx.all_pairs_dijkstra_path_length(G, cutoff=10e-8)
   
    for r in results:

        source = r[0]
        distances = r[1]
        
        sorted_distances = np.array(sorted(distances.items(), key=lambda item: item[1]))

        k0 = min(k, sorted_distances.shape[0])

        nearest_neighbours[source,:k0] = sorted_distances[:k0,0]
        nearest_distances[source,:k0] = sorted_distances[:k0,1]
        
    # normalize distances
    min_val = np.min(nearest_distances)
    max_val = np.max(nearest_distances)
    norm_dij = (nearest_distances - min_val) / (max_val - min_val) 

    return nearest_neighbours, norm_dij


def calculate_ged(adj_uniq,k=100):

    '''
    Uses the adjacency matrix representation of the secondary structures
    to calculate a distance proportional to the number of elementary steps.

    dist(x,y) = 2 * min. no. elementary steps separating x and y
 
    Returns: 
        nearest_neighbours[i]: 
            indices of the k states nearest to state i (ties broken arbitrarily)
        nearest_distances[i][k]: 
            distance between state i and state nearest_distances[i][k]
    '''

    # TODO: remove the 2x factor. Rescale loss as necessary 

    n_states = len(adj_uniq)
    n_bases = len(adj_uniq[0])

    k = min(k, n_states) # TODO: add warning for k > n_states

    annoy_index = AnnoyIndex(n_bases**2, 'manhattan')

    # Add vectors to the index
    for i in range(n_states):
        annoy_index.add_item(i, adj_uniq[i].flatten())

    # TODO: parameter tuning
    #       set n_trees as large as possible within memory constraints
    #       set search_k as large as possible within time constraints

    # Build the index
    annoy_index.build(n_trees=10, n_jobs=-1) 

    nearest_neighbours = np.zeros((n_states,k))
    nearest_distances = np.zeros((n_states,k))

    for i in tqdm.tqdm(range(n_states)):
        indices, distances = annoy_index.get_nns_by_item(i, k, search_k=-1, include_distances=True)

        nearest_neighbours[i,:] = indices
        nearest_distances[i,:] = distances

    return nearest_neighbours, nearest_distances


def calculate_prob(dp_og_uniq, id_uniq, sequences):

    '''
    Returns: prop[i] = Nupack equilibrium probability of state i
    '''
  
    # TODO: add celsius, sodium, magnesium to raw data file and use in Nupack model
    eq_model = nu.Model(material='dna04-nupack3')

    n_states = dp_og_uniq.shape[0]
    prob = np.zeros(n_states, dtype=float)   

    for i in tqdm.tqdm(range(n_states)):

        strcts = dp_og_uniq[i].split(" ")
        seqs = [[sequences[s] for s in id_uniq[i][j]] for j in range(len(id_uniq[i]))]
       
        prob[i] = np.prod([nu.structure_probability(strands=seqs[j], structure=strcts[j], model=eq_model) for j in range(len(seqs))])
       
    return prob


# TODO: 
# Think about the consequences of having a cutoff at a constant 100 nodes, rather than value-based cuttof 
#   - Some states may have many more nearby states than others
#   - there is quite a bit of arbitrary tie-breaking in calculate_ged(). Are results robust to ways of breaking ties?
# Think about the consequences of a state s being duplicated several times as its own nearest neighbour (in calculate_mpt) 
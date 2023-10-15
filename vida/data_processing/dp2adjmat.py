import numpy as np
import networkx as nx
import pickle

import argparse


# convert dot-parenthesis notation to adjacency matrix
def dot2adj(db_str,hairpin=False,helix=True):
    """converts DotBracket str to np adj matrix
    
    Args:
        db_str (str): N-len dot bracket string
    
    Returns:
        [np array]: NxN adjacency matrix
    """
    
    dim = len(str(db_str))

    # get pair tuples
    pair_list = dot2pairs(db_str)
    sym_pairs = symmetrized_edges(pair_list)


    # initialize the NxN mat (N=len of RNA str)
    adj_mat = np.zeros((dim,dim))

    adj_mat[sym_pairs[0,:], sym_pairs[1,:]] = 1
    
    if hairpin == True:
        True

    if helix == True:
        assert dim % 2 == 0, "Not a valid helix sequence."
        end2head = np.ceil(dim/2).astype(int)

        if db_str[end2head-1:end2head+1] != "()":
            adj_mat[end2head-1, end2head] = 0
            adj_mat[end2head, end2head-1] = 0

    return adj_mat

def dot2pairs(dp_str):
    """converts a DotBracket str to adj matrix

    uses a dual-checking method
    - str1 = original str
    - str2 = reversed str

    iterates through both strings simult and collects indices
    forward str iteration: collecting opening indicies - list1
    backwards str iteration: collecting closing indices - list2
    - as soon as a "(" is found in str2, the first(typo,last?) entry of list1 is paired
      with the newly added index/entry of list2 
    
    Args:
        dotbracket_str (str): dot bracket string (ex. "((..))")
    
    Returns:
        [array]: numpy adjacency matrix
    """ 
    dim = len(str(dp_str))

    # pairing indices lists
    l1_indcs = []
    l2_indcs = []
    pair_list = []

    for indx in range(dim):
        
        # checking stage
        # forward str
        if dp_str[indx] == "(":

            l1_indcs.append(indx)
 
        if dp_str[indx] == ")":
            l2_indcs.append(indx)

        # pairing stage
        # check that either list is not empty
        if len(l2_indcs) * len(l1_indcs) > 0:
            pair = (l1_indcs[-1], l2_indcs[0])
            pair_list.append(pair)
        
            # cleaning stage
            l1_indcs.pop(-1)
            l2_indcs.pop(0)
    
    # get path graph pairs
    G = nx.path_graph(dim)
    path_graph_pairs = G.edges()
    
    return pair_list + list(path_graph_pairs)

def symmetrized_edges(pairs_list):
    
    # conver pairs to numpy array [2,-1]
    edge_array = np.array(pairs_list)
 
    # concatenate with opposite direction edges
    # print(edge_array.T[[1,0]].T.shape)
    reverse_edges = np.copy(edge_array)
    reverse_edges[:, [0,1]] = reverse_edges[:, [1,0]]
    full_edge_array = np.vstack((edge_array, reverse_edges))
    
    return full_edge_array.T

# convert dot-parenthesis notation to adjacency matrix in a single trajectory
def sim_adj(dps):
    """convert dot-parenthesis notation to adjacency matrix
    Args:
        sim: [list of sims] dot-parenthesis notation, energy floats
            eg. ['...............', ...]
    Returns:
        (tuple): NxN adjacency np matrix
    """
    adj_mtr = []
        
    for s in dps:
        adj = dot2adj(s)
        adj_mtr.append(adj)
    adj_mtr = np.array(adj_mtr) # get adjacency matrix

    return adj_mtr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', required=True, help='preprocessed data file')
    parser.add_argument('--outpath', required=True, help='output adjacency matrix')

    args = parser.parse_args()

    inpath = args.inpath
    outpath = args.outpath
    
    # Load the data
    print(f"[dp2adj] Loading preprocessed SIMS_dp_uniq from {inpath}")
    
    with open(inpath, 'rb') as file:
        loaded_data = pickle.load(file)
    
    SIMS_dp_uniq = loaded_data["SIMS_dp_uniq"]
    
    # convert dot-parenthesis notation to adjacency matrix
    print("[dp2adj] Converting dot-parenthesis notation to adjacency matrix")
    
    SIMS_adj_uniq = sim_adj(SIMS_dp_uniq)
    
    # save adjacency matrix
    print(f"[dp2adj] Saving adjacency matrix to {outpath}")
 
    data_to_save = {
    "SIMS_adj_uniq": SIMS_adj_uniq,
    }
    
    with open(outpath, 'wb') as file:
        pickle.dump(data_to_save, file) 
        
    print("[dp2adj] Done!")
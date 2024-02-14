import numpy as np
import networkx as nx
import re
from itertools import permutations



############### Two-Strand Structure ###############
####################################################

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




############### Three-Strand Structure ###############
####################################################

# find strand permutation that matches the sequence 
# then get the corresponding name list
def concat_disorder(trajs_seq, ref_name_list, strand_list):
    alter_name_list = []
    
    for i in range(len(trajs_seq)):
        sequence_list = re.split(r'\s|\+', trajs_seq[i]) 
        sequence = ''.join(sequence_list)
        found_match = False
        for permuted_strand in permutations(strand_list):
            combined_sequence = ''.join(permuted_strand)
            if combined_sequence == sequence:
                curr_name_list = [ref_name_list[strand_list.index(strand)] for strand in permuted_strand]
                alter_name_list.append(curr_name_list)                
                found_match = True
                break
        
        if not found_match:
            print('Error: sequence not found')
            
            
    return np.array(alter_name_list,dtype=object)



# convert dot-parenthesis notation to adjacency matrix for three-strand
def dp2adj_3strand(ref_name, alter_name, alter_name_arr, dp_structure):
    # construct backbone edges
    def build_consecutive_edges(input_list):
        edges = [(input_list[i], input_list[i+1]) for i in range(len(input_list)-1)]
        
        return edges


    # build backbone edges
    backbones = []
    
    for base_names_strand in alter_name_arr:
        backbone = build_consecutive_edges(base_names_strand)
        backbones.extend(backbone)
    
    
    # build base pair edges
    stack = []  # Initialize stack to keep track of opening brackets
    base_pairs = []  # Initialize list to store pairs    
    
    for name, char in zip(alter_name, dp_structure):
        
        if char == '(':
            stack.append(name)  # Push index of opening bracket onto stack
        elif char == ')':
            if stack:
                opening_index = stack.pop()  # Pop top index from stack
                base_pairs.append((opening_index, name))  # Create a pair
            else:
                print("Error: Mismatched brackets")
                return None
    
    if stack:
        print("Error: Mismatched brackets")
        return None
    
    
    # collect all edges
    all_pairs = backbones + base_pairs
 
    # assign nodes and edges
    nodes = ref_name 
    edges = all_pairs 

    # Initialize adjacency matrix with zeros
    adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

    # Populate the adjacency matrix based on edges
    for edge in edges:
        i = nodes.index(edge[0])
        j = nodes.index(edge[1])
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

    return adjacency_matrix



def sim_adj_3strand_uniq(dp_arr, trajs_seqs, ref_name, ref_name_list, strand_list, indices_uniq):
    
    adj_uniq = np.zeros((len(indices_uniq), len(ref_name), len(ref_name)), dtype=int)
    seqlabel_uniq = np.full(len(indices_uniq), fill_value="", dtype=object)
    counter = 0

    for trjID, dp_structures in enumerate(dp_arr):
        alter_name_arr = concat_disorder(trajs_seqs[trjID],ref_name_list, strand_list)
            
        for i, dp in enumerate(dp_structures):
            if counter in indices_uniq:
                position = np.where(indices_uniq == counter)[0][0]
                alter_name = np.concatenate(alter_name_arr[i])
                adjmtrx = dp2adj_3strand(ref_name, alter_name, alter_name_arr[i], dp)
                adj_uniq[position] = adjmtrx
                
                seqlabel = trajs_seqs[trjID][i]
                seqlabel_uniq[position] = seqlabel                
                
            counter += 1
                
    return adj_uniq, seqlabel_uniq


    
def sim_adj_3strand(dp_arr, trajs_seqs, ref_name, ref_name_list, strand_list):
    
    adj_mtr = []
    
    for trjID, dp_structures in enumerate(dp_arr):
        alter_name_arr = concat_disorder(trajs_seqs[trjID],ref_name_list, strand_list)
                
        for i, dp in enumerate(dp_structures):
            alter_name = np.concatenate(alter_name_arr[i])

            adj_mtr.append(dp2adj_3strand(ref_name, alter_name, alter_name_arr[i], dp))
    
    return np.array(adj_mtr)

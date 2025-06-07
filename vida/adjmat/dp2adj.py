import numpy as np
import tqdm


############### Three-Strand Structure ###############
####################################################

def reorder_base_names(order_id, base_names):
    
    return np.array([base for cmplx in order_id for strand in cmplx for base in base_names[strand]])


# convert dot-parenthesis notation to undirected graph (adjacency matrix representation)
def dp2adj_3strand(base_names_reordered, dp_structure, nodes):

    # TODO: consider treating backbone and bp edges differently. Maybe backbone edges should be directed, ie 3'->5'? 

    # build backbone edges
    backbones = [(base_names_reordered[i],base_names_reordered[i+1]) for i in range(len(base_names_reordered)-1) if base_names_reordered[i][0]==base_names_reordered[i+1][0]]

    # build base pair edges
    stack = []  # Initialize stack to keep track of opening brackets
    base_pairs = []  # Initialize list to store pairs    
    
    for base, char in zip(base_names_reordered, dp_structure):
        
        if char == '(':
            stack.append(base)  # Push index of opening bracket onto stack
        elif char == ')':
            if stack:
                other_base = stack.pop()  # Pop top index from stack
                base_pairs.append((other_base, base))  # Create a pair
            else:
                print("Error: Mismatched brackets")
                return None
    
    if stack:
        print("Error: Mismatched brackets")
        return None
    
    # collect all edges
    edges = backbones + base_pairs

    # Initialize adjacency matrix with zeros
    adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

    # Populate the adjacency matrix based on edges
    for edge in edges:
        i = nodes.index(edge[0])
        j = nodes.index(edge[1])
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1

    return adjacency_matrix

    

def construct_adj_matrices(dps, orders, base_names):
    
    nodes = sorted(base for strand in base_names for base in strand)

    adj_matrices = []

    for dp, order_id in tqdm.tqdm(zip(dps, orders), total=len(dps)):
        base_names_reordered = reorder_base_names(order_id, base_names)
        adj_matrices.append(dp2adj_3strand(base_names_reordered, dp, nodes))
    
    return np.array(adj_matrices)
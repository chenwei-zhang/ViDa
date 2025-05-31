import numpy as np
import tqdm


############### Three-Strand Structure ###############
####################################################

def solve_order(order_id, ref_name_list):
    """ 
    0 refers to the incumbent (33), denoted as "a"
    1 refers to the substrate (27), denoted as "b"
    2 refers to the invader (24),   denoted as "c"
    default ref_name_list order: a, b, c 
    """
    
    if order_id == [(0, 1, 2)] or order_id == [(0, 1),(2,)] or order_id == [(0,),(1, 2)]:
        alter_name = np.concatenate([ref_name_list[0], ref_name_list[1], ref_name_list[2]])
    
    elif order_id == [(0, 2, 1)] or order_id == [(0, 2),(1,)] or order_id == [(0,),(2, 1)]:
        alter_name = np.concatenate([ref_name_list[0], ref_name_list[2], ref_name_list[1]])
                
    else:
        print(order_id)
        raise ValueError("Invalid reaction ordering")
    
    return alter_name



# convert dot-parenthesis notation to adjacency matrix for three-strand
def dp2adj_3strand(ref_name, alter_name, dp_structure):
    # construct backbone edges
    def build_consecutive_edges(input_list):
        edges = [(input_list[i], input_list[i+1]) for i in range(len(input_list)-1)]
        
        return edges

     # build backbone edges
    all_backbone_edges = build_consecutive_edges(alter_name)
    # remove cross-strand edges
    backbones = []
    for edge in all_backbone_edges:
        # Only keep edges that connect nucleotides with the same prefix (a-a, b-b, c-c)
        if edge[0][0] == edge[1][0]:
            backbones.append(edge)
            
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
    nodes = ref_name.tolist() 
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

    

def sim_adj_3strand(dps, orders, ref_name_list):
    
    ref_name = np.concatenate([ref_name_list[0], ref_name_list[1], ref_name_list[2]])
    adj_mtr = []
    
    for dp, order_id in tqdm.tqdm(zip(dps, orders), total=len(dps)):
        alter_name = solve_order(order_id, ref_name_list)
        adj_mtr.append(dp2adj_3strand(ref_name, alter_name, dp))
    
    return np.array(adj_mtr)
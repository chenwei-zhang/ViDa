import numpy as np
import argparse
import time
from dp2adj import construct_adj_matrices


if __name__ == '__main__':
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--rxn', required=True, help='Reaction name')
        
    args = parser.parse_args()

    reaction_id = args.rxn

    inpath = "data/post_data/{}/preprocess_{}.npz".format(reaction_id, reaction_id)
    outpath = "data/post_data/{}/adjmat_{}.npz".format(reaction_id, reaction_id)

    
    # Load the data
    print(f"[dp2adj] Loading preprocessed dp_uniq from {inpath}")
    print("[dp2adj] Loading preprocessed")
    print(f"[dp2adj] Converting dot-parenthesis notation to adjacency matrix")
    
    loaded_data = np.load(inpath, allow_pickle=True)
        
    base_names = loaded_data["base_names"]
    dp_uniq = loaded_data["dp_uniq"]
    id_uniq = loaded_data["id_uniq"]
    # convert dot-parenthesis notation to adjacency matrix
    adj_uniq = construct_adj_matrices(dp_uniq, id_uniq, base_names)    
                        
    # save adjacency matrix
    print(f"[dp2adj] Saving adjacency matrix to {outpath}")
 
    data_to_save = {
        "adj_uniq": adj_uniq,
    }
    
    np.savez_compressed(outpath, **data_to_save)
    
    print("[dp2adj] Done!")
    
    # Record the end time
    end_time = time.time()
    
    # Print the time elapsed
    print(f"[dp2adj] Time elapsed: {(end_time - start_time):.3f} seconds")
import numpy as np
import argparse
import time
from dp2adj import construct_adj_matrices


if __name__ == '__main__':
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', required=True, help='preprocessed data file')
    parser.add_argument('--outpath', required=True, help='output adjacency matrix')
    parser.add_argument('--num_strand', type=int, default=3, help='number of strands')
        
    args = parser.parse_args()

    inpath = args.inpath
    num_strand = args.num_strand
    outpath = args.outpath
    
    
    # Load the data
    print(f"[dp2adj] Loading preprocessed dp_uniq from {inpath}")
    print("[dp2adj] Loading preprocessed")
    print(f"[dp2adj] Number of strands: {num_strand}")
    print(f"[dp2adj] Converting dot-parenthesis notation to adjacency matrix")
    
    loaded_data = np.load(inpath, allow_pickle=True)
        
    if num_strand == 3:
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
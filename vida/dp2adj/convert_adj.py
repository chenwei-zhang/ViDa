import numpy as np
import argparse
import time
from dp2adj import sim_adj, sim_adj_3strand


if __name__ == '__main__':
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', required=True, help='preprocessed data file')
    parser.add_argument('--num_strand', required=True, help='number of strands')
    parser.add_argument('--outpath', required=True, help='output adjacency matrix')

    args = parser.parse_args()

    inpath = args.inpath
    num_strand = args.num_strand
    outpath = args.outpath
    
    # Load the data
    print(f"[dp2adj] Loading preprocessed dp_uniq from {inpath}")
    
    loaded_data = np.load(inpath)
    
    
    # convert dot-parenthesis notation to adjacency matrix
    print("[dp2adj] Converting dot-parenthesis notation to adjacency matrix")
    
    if num_strand == "2":
        dp_uniq = loaded_data["dp_uniq"]
        adj_uniq = sim_adj(dp_uniq)
        
    elif num_strand == "3":
        dp_arr, trajs_seqs, ref_name, ref_name_list, strand_list = loaded_data["dp_arr"], loaded_data["trajs_seqs"], loaded_data["ref_name"], loaded_data["ref_name_list"], loaded_data["strand_list"] TODO
        adj_uniq = sim_adj_3strand(dp_arr, trajs_seqs, ref_name, ref_name_list, strand_list)  
        
          
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
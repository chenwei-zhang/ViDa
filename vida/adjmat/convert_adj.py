import numpy as np
import argparse
import time
import pickle
import gzip
from dp2adj import sim_adj, sim_adj_3strand_uniq


if __name__ == '__main__':
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', required=True, help='preprocessed data file')
    parser.add_argument('--outpath', required=True, help='output adjacency matrix')
    parser.add_argument('--num-strand', type=int, default=2, help='number of strands')

    if parser.parse_known_args()[0].num_strand == 3:
        parser.add_argument('--seq-path', type=str, required=True, help="sequence file path")
        
    args = parser.parse_args()

    inpath = args.inpath
    num_strand = args.num_strand
    outpath = args.outpath
    
    
    # Load the data
    print(f"[dp2adj] Loading preprocessed dp_uniq from {inpath}")
    
    
    
    # convert dot-parenthesis notation to adjacency matrix
    print("[dp2adj] Loading preprocessed")
    print(f"[dp2adj] Number of strands: {num_strand}")
    print(f"[dp2adj] Converting dot-parenthesis notation to adjacency matrix")
    
    if num_strand == 2:
        loaded_data = np.load(inpath, allow_pickle=True)
        
        dp_uniq = loaded_data["dp_uniq"]
        adj_uniq = sim_adj(dp_uniq)
        
    elif num_strand == 3:
        # load nessary data for graph construction
        seq_path = args.seq_path
        
        # Load the data
        with gzip.open(seq_path, 'rb') as file:
            load_data_seq = pickle.load(file)
        trajs_seqs = load_data_seq["trajs_seqs"]
        ref_name = load_data_seq["ref_name"]
        ref_name_list = load_data_seq["ref_name_list"]
        strand_list = load_data_seq["strand_list"]
        
        loaded_data = np.load(inpath, allow_pickle=True)
        dp_arr = loaded_data["dp_arr"]
        indices_uniq = loaded_data["indices_uniq"]
        
        adj_uniq = sim_adj_3strand_uniq(dp_arr, trajs_seqs, ref_name, ref_name_list, strand_list, indices_uniq)    
                            
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
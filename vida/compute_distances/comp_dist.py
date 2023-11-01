import numpy as np
import argparse
import time
from mpt_ged import get_all_edges, build_wdg, calculate_mpt, calculate_ged, calculate_prob


if __name__ == '__main__':
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', required=True, help='preprocessed data, time data')
    parser.add_argument('--holdtime', required=True, help='average holding time for each node')
    parser.add_argument('--adjmat', required=True, help='ajacency matrix for each node')
    parser.add_argument('--outpath', required=True, help='output minimum passage time distance')

    args = parser.parse_args()

    inpath = args.inpath
    holdtime = args.holdtime
    outpath = args.outpath
    adjmat = args.adjmat
    
    # Load the data
    print(f"[Comp_dist] Loading preprocessed index from {inpath}")
    
    loaded_data = np.load(inpath)
    
    indices_all = loaded_data["indices_all"]
        
    
    print(f"[Comp_dist] Loading average holding time from {holdtime}")

    loaded_data = np.load(holdtime)
        
    hold_time_uniq = loaded_data["hold_time_uniq"]
    trj_id = loaded_data["trj_id"]
    
    
    print(f"[Comp_dist] Loading adjacency matrix from {adjmat}")

    loaded_data = np.load(adjmat)
    
    adj_uniq = loaded_data["adj_uniq"]


    # Build the edges
    print("[Comp_dist] Building the edges")
    
    all_edges = get_all_edges(indices_all, trj_id)
    
    
    # Construct the modified weighted undirected graph
    print("[Comp_dist] Constructing the weighted directed graph")
    
    DG = build_wdg(all_edges, hold_time_uniq)
    
    
    # Calculate the graph edit distance between X_i and x_dj
    print("[Comp_dist] Computing the minimum passage time distance")
    
    x_dj, d_ij = calculate_mpt(DG)
    
    
    # Calculate the graph edit distance between X_i and x_ej
    print("[Comp_dist] Computing the graph edit distance")
    
    x_ej, e_ij = calculate_ged(adj_uniq)
    
    
    # Calculate the probability of being visited during a simulated trajectory
    print("[Comp_dist] Computing the node probability")
    
    p_i = calculate_prob(indices_all, trj_id, hold_time_uniq)
    
    
    # save pickle file for shortest path
    print(f"[Comp_dist] Saving MPT and GED to {outpath}")
    
    data_to_save = {
    "x_dj": x_dj,
    "x_ej": x_ej,
    "d_ij": d_ij,
    "e_ij": e_ij,
    "p_i": p_i,
    }
    
    np.savez_compressed(outpath, **data_to_save)
        
    print("[Comp_dist] Done!")
    
    # Record the end time
    end_time = time.time()
    
    # Print the time elapsed
    print(f"[Comp_dist] Elapsed Time: {(end_time - start_time):.3f} seconds")
    

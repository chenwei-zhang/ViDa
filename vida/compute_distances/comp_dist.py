import numpy as np
import argparse
import time
from mpt_ged import get_transitions, build_wdg, calculate_mpt, calculate_ged, calculate_prob
import pandas as pd


def load_raw_data(data_filename, name): 
   
    df = pd.read_csv(data_filename)

    row = df[df['reaction_id'] == name].index[0]
        
    row_data = df.loc[row]

    sequences = {"incumbent": row_data["incumbent"],
                 "substrate": row_data["substrate"],
                 "invader": row_data["invader"]}

    return sequences

if __name__ == '__main__':
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--rxn', required=True, help='Reaction name')

    args = parser.parse_args()

    reaction_id = args.rxn

    inpath = "data/post_data/{}/preprocess_{}.npz".format(reaction_id, reaction_id)
    holdtime = "data/post_data/{}/time_{}.npz".format(reaction_id, reaction_id)
    adjmat = "data/post_data/{}/adjmat_{}.npz".format(reaction_id, reaction_id)
    outpath = "data/post_data/{}/mpt-ged_{}.npz".format(reaction_id, reaction_id)


    # Load the data
    print(f"[Comp_dist] Loading preprocessed index from {inpath}")
    
    loaded_data = np.load(inpath, allow_pickle=True)
    
    indices_all = loaded_data["indices_all"]

    dp_og_uniq = loaded_data['dp_og_uniq']
    id_uniq = loaded_data['id_uniq']
    
    print(f"[Comp_dist] Loading average holding time from {holdtime}")

    loaded_data = np.load(holdtime)
        
    hold_time_uniq = loaded_data["hold_time_uniq"]
    endpoints = loaded_data["trj_id"]
    
    
    print(f"[Comp_dist] Loading adjacency matrix from {adjmat}")

    loaded_data = np.load(adjmat)
    
    adj_uniq = loaded_data["adj_uniq"]


    # Build the edges
    print("[Comp_dist] Building the edges")
    
    transitions = get_transitions(indices_all, endpoints) 
    
    
    # Construct the modified weighted undirected graph
    print("[Comp_dist] Constructing the weighted directed graph")
    
    DG = build_wdg(transitions, hold_time_uniq) 
    
    
    # Calculate the graph edit distance between X_i and x_dj
    print("[Comp_dist] Computing the minimum passage time distance")
    
    x_dj, d_ij = calculate_mpt(DG) 
    
    
    # Calculate the graph edit distance between X_i and x_ej
    print("[Comp_dist] Computing the graph edit distance")
    
    x_ej, e_ij = calculate_ged(adj_uniq)
    
    
    # Calculate the probability of being visited during a simulated trajectory
    print("[Comp_dist] Computing the node probability")
    
    sequences = load_raw_data("data/raw_data.csv", reaction_id)
    sequences_list = [sequences['incumbent'],sequences['substrate'],sequences['invader']]
    p_i = calculate_prob(dp_og_uniq, id_uniq, sequences_list)
    
    
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
    

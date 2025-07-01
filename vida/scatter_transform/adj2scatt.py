import numpy as np
import argparse
import time
from gsae.scattering.scattering import transform_dataset, get_normalized_moments


if __name__ == '__main__':
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--rxn', required=True, help='Reaction name')

    args = parser.parse_args()
    reaction_id = args.rxn
        
    inpath = "data/post_data/{}/adjmat_{}.npz".format(reaction_id, reaction_id)
    outpath = "data/post_data/{}/scatt_{}.npz".format(reaction_id, reaction_id)

    # Load the data
    print(f"[adj2scatt] Loading preprocessed adj_uniq from {inpath}")
    
    loaded_data = np.load(inpath)
    
    adj_uniq = loaded_data["adj_uniq"]
    
    # # Multiple trajectories
    print("[adj2scatt] Converting adjacency matrix to scattering coefficients")
    
    scat_coeff_array = transform_dataset(adj_uniq)
    scar_uniq = get_normalized_moments(scat_coeff_array).squeeze()
    
    print(f"[adj2scatt] Saving scattering coefficients to {outpath}")
    
    data_to_save = {
    "scar_uniq": scar_uniq,
    }
    
    np.savez_compressed(outpath, **data_to_save)
    
    print("[adj2scatt] Done!")
    
    # Record the end time
    end_time = time.time()
    
    # Print the time elapsed    
    print(f"[adj2scatt] Elapsed Time: {(end_time - start_time):.3f} seconds")
    
    
    
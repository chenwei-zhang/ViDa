import numpy as np
import pickle
import gzip
import os
import argparse
import time
from utils import concat_machinek, get_uniq


def main():
    parser = argparse.ArgumentParser(description='Load Data')
    parser.add_argument('--inpath', required=True, help='Path to input pickle data file')
    parser.add_argument('--outpath', required=True, help='Output pickle file path')
    
    args = parser.parse_args()

    inpath = args.inpath
    outpath = args.outpath
    
    print(f"[Preprocess] Loading data from {inpath}")
    
    # Load the data
    with gzip.open(inpath, 'rb') as file:
        loaded_data = pickle.load(file)

    # Get the data from the pickle file 
    states = loaded_data["trajs_states"]
    times = loaded_data["trajs_times"]
    energies = loaded_data["trajs_energies"]
        
    print("[Preprocess] Preprocess Machinek data")
    ref_name_list = loaded_data["ref_name_list"]
    trajs_ids = loaded_data["trajs_ids"]
    dp, dp_og, energy, trans_time, order_cid = concat_machinek(states, times, energies, trajs_ids)

    # get the unique structures and their corresponding indices
    print("[Preprocess] Get the unique structures and their corresponding indices")
    dp_uniq, dp_og_uniq, energy_uniq, id_uniq, _, indices_uniq, indices_all = get_uniq(dp, dp_og, energy, order_cid)
    
    # save read data
    print(f"[Preprocess] Saving preprocessed data to {outpath}")
    data_to_save = {
    "dp_uniq": dp_uniq,
    "dp_og_uniq": dp_og_uniq,
    "energy_uniq": energy_uniq,
    "id_uniq": id_uniq,
    "indices_uniq": indices_uniq,
    "indices_all": indices_all,
    "trans_time": trans_time,
    "ref_name_list": np.array(ref_name_list, dtype=object),
    }
    
    # save the data to npz file
    np.savez_compressed(outpath, **data_to_save)

    print("[Preprocess] Done!")
        

if __name__ == '__main__':
    # Record the start time
    start_time = time.time()
    
    main()
    
    # Record the end time
    end_time = time.time()
    
    # Print the elapsed time
    print(f"[Preprocess] Elapsed Time: {(end_time - start_time):.3f} seconds")
    
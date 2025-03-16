import numpy as np
import pickle
import gzip
import os
import argparse
import time
from utils import concat_hata, concat_gao, concat_machinek, get_uniq


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
    
    file_name = os.path.basename(inpath).lower()
    
    if "hata" in file_name:
        print("[Preprocess] Preprocess Hata data")
        
        type_uniq = loaded_data["trajs_types"]
        dp, dp_og, pair, energy, trans_time = concat_hata(states, times, energies)
        
    elif "gao" in file_name:
        print("[Preprocess] Preprocess Gao data")
        
        pairs = loaded_data["trajs_pairs"]
        dp, dp_og, pair, energy, trans_time = concat_gao(states, times, energies, pairs)

    else:
        print("[Preprocess] Preprocess Machinek data")

        ref_name = loaded_data["ref_name"]
        dp, dp_og, energy, trans_time = concat_machinek(states, times, energies)
        pair = None
        
    # else:
    #     print("Wrong file name")


    # get the unique structures and their corresponding indices
    print("[Preprocess] Get the unique structures and their corresponding indices")
    
    dp_uniq, dp_og_uniq, energy_uniq, pair_uniq, indices_uniq, indices_all = get_uniq(dp, dp_og, energy, pair)
    
    # save read data
    print(f"[Preprocess] Saving preprocessed data to {outpath}")
    
    data_to_save = {
    "dp_uniq": dp_uniq,
    "dp_og_uniq": dp_og_uniq,
    "energy_uniq": energy_uniq,
    "indices_uniq": indices_uniq,
    "indices_all": indices_all,
    "trans_time": trans_time,
    }
    
    if "hata" in file_name:
        data_to_save["type_uniq"] = type_uniq
        data_to_save["pair_uniq"] = pair_uniq
        
    elif "gao" in file_name:
        data_to_save["pair_uniq"] = pair_uniq
    
    else:
        data_to_save["ref_name"] = ref_name

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
    
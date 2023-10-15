import argparse
import numpy as np
import pickle
import os
from utils import concat_hata, concat_gao, get_uniq


def main():
    parser = argparse.ArgumentParser(description='Load Data')
    parser.add_argument('--inpath', required=True, help='Path to input pickle data file')
    parser.add_argument('--outpath', required=True, help='Output pickle file path')
    
    args = parser.parse_args()

    inpath = args.inpath
    outpath = args.outpath
    
    print(f"[Preprocess] Loading data from {inpath}")
    
    # Load the data
    with open(inpath, 'rb') as file:
        loaded_data = pickle.load(file)

    # Get the data from the pickle file 
    states = loaded_data["trajs_states"]
    times = loaded_data["trajs_times"]
    energies = loaded_data["trajs_energies"]
    
    file_name = os.path.basename(inpath)
    
    if "Hata" in file_name:
        print("[Preprocess] Preprocess Hata data")
        
        SIMS_type_uniq = loaded_data["trajs_types"]
        SIMS_dp, SIMS_dp_og, SIMS_pair, SIMS_G, SIMS_T = concat_hata(states, times, energies)
        
    elif "Gao" in file_name:
        print("[Preprocess] Preprocess Gao data")
        
        pairs = loaded_data["trajs_pairs"]
        SIMS_dp, SIMS_dp_og, SIMS_pair, SIMS_G, SIMS_T = concat_gao(states, times, energies, pairs)
    
    else:
        print("Wrong file name")
        return

    # get the unique structures and their corresponding indices
    print("[Preprocess] Get the unique structures and their corresponding indices")

    SIMS_dp_uniq, SIMS_dp_og_uniq, SIMS_pair_uniq, SIMS_G_uniq, indices_S, coord_id_S = get_uniq(SIMS_dp, SIMS_dp_og, SIMS_pair, SIMS_G)
    
    # save read data
    print(f"[Preprocess] Saving preprocessed data to {outpath}")
    
    data_to_save = {
    "SIMS_dp_uniq": SIMS_dp_uniq,
    "SIMS_dp_og_uniq": SIMS_dp_og_uniq,
    "SIMS_pair_uniq": SIMS_pair_uniq,
    "SIMS_G_uniq": SIMS_G_uniq,
    "indices_S": indices_S,
    "coord_id_S": coord_id_S,
    "SIMS_T": SIMS_T,
    }
    
    if "Hata" in file_name:
        data_to_save["SIMS_type_uniq"] = SIMS_type_uniq
    
    # Save the data to the file using pickle
    with open(outpath, 'wb') as file:
        pickle.dump(data_to_save, file)
        
    print("[Preprocess] Done!")
        

if __name__ == '__main__':
    main()
    
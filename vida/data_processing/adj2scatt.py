import sys
sys.path.append('/Users/chenwei/Desktop/Github/ViDa') 

import numpy as np
import pickle
import argparse
from vida.scatter.scatter_transform import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', required=True, help='preprocessed data file')
    parser.add_argument('--outpath', required=True, help='output adjacency matrix')

    args = parser.parse_args()

    inpath = args.inpath
    outpath = args.outpath
        
    # Load the data
    print(f"[adj2scatt] Loading preprocessed SIMS_adj_uniq from {inpath}")
    
    with open(inpath, 'rb') as file:
        loaded_data = pickle.load(file)
    
    SIMS_adj_uniq = loaded_data["SIMS_adj_uniq"]
    
    # # Multiple trajectories
    print("[adj2scatt] Converting adjacency matrix to scattering coefficients")
    
    scat_coeff_array_S = transform_dataset(SIMS_adj_uniq)
    SIMS_scar_uniq = get_normalized_moments(scat_coeff_array_S).squeeze()
    
    print(f"[adj2scatt] Saving scattering coefficients to {outpath}")
    
    data_to_save = {
    "SIMS_scar_uniq": SIMS_scar_uniq,
    }
    
    with open(outpath, 'wb') as file:
        pickle.dump(data_to_save, file)
    
    print("[adj2scatt] Done!")
    
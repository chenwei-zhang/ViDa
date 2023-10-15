import argparse
import numpy as np
import pickle
from utils import read_Gao

def main():
    parser = argparse.ArgumentParser(description='Load Data')
    parser.add_argument('--inpath', required=True, help='Path to input data file')
    parser.add_argument('--rxn', required=True, help='Reaction name')
    parser.add_argument('--outpath', required=True, help='output file path')
    
    args = parser.parse_args()

    inpath = args.inpath
    rxn = args.rxn
    outpath = args.outpath

    # Load data
    trajs_states, trajs_times, trajs_energies, trajs_pairs = read_Gao(inpath, rxn)

    # save read data
    fsave = f'{outpath}/{rxn}.pkl.gz'

    data_to_save = {
    "trajs_states": trajs_states,
    "trajs_times": trajs_times,
    "trajs_energies": trajs_energies,
    "trajs_pairs": trajs_pairs,
    }
    
    # Save the data to the file using pickle
    with open(fsave, 'wb') as file:
        pickle.dump(data_to_save, file)
        
        
if __name__ == '__main__':
    main()
    

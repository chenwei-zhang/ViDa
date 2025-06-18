import os
import argparse
import pickle
import gzip
import time

from utils import read_machinek, assign_base_names, load_raw_data



def main():
    parser = argparse.ArgumentParser(description='Load Data')
    parser.add_argument('--inpath', required=True, help='Path to input data file')
    parser.add_argument('--rxn', required=True, help='Reaction name')
    parser.add_argument('--num_traj', required=True, type=int, help='Number of files')
    parser.add_argument('--outpath', required=True, help='output file path')
    
    args = parser.parse_args()

    inpath = args.inpath
    reaction_id = args.rxn
    outpath = args.outpath
    num_traj = args.num_traj

    sequences = load_raw_data("raw_data.csv", reaction_id)
       
    base_names = assign_base_names(sequences['incumbent'], sequences['substrate'], sequences['invader'])
    
    # Load data
    fpath = os.path.join(inpath, f"{reaction_id}.hdf5")
    print(f"[Read] Loading data from {fpath}")
    
    trajs_states,trajs_times,trajs_energies,trajs_ids = read_machinek(
                                                    fpath,
                                                    num_traj
                                                    )
    
    # save read data
    print(f"[Read] Saving preprocessed data to {outpath}")
    
    outpath_dir = os.path.dirname(outpath)
    print(outpath_dir)
    os.makedirs(outpath_dir, exist_ok=True)
    
    data_to_save = {
    "trajs_states": trajs_states,
    "trajs_times": trajs_times,
    "trajs_energies": trajs_energies,
    "trajs_ids": trajs_ids,
    "base_names": base_names,
    }
    
    # Save the data to the file using pickle
    with gzip.open(outpath, 'wb') as file:
        pickle.dump(data_to_save, file)
    
    print("[Read] Done!")
        
        
        
        
if __name__ == '__main__':
    # Record the start time
    start_time = time.time()
    
    main()
    
    # Record the end time
    end_time = time.time()
    
    # Print the time elapsed
    print(f"[Read] Elapsed Time: {(end_time - start_time):.3f} seconds")
    

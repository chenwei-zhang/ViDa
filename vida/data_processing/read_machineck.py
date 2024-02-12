import argparse
import numpy as np
import pickle
import gzip
from utils import read_machinek, assign_base_names
import time


def main():
    parser = argparse.ArgumentParser(description='Load Data')
    parser.add_argument('--inpath', required=True, help='Path to input data file')
    parser.add_argument('--rxn', required=True, help='Reaction name')
    parser.add_argument('--outpath', required=True, help='output file path')
    
    args = parser.parse_args()

    inpath = args.inpath
    rxn = args.rxn
    outpath = args.outpath

    if rxn == "Machinek-PRF":
        ref_strands = 'CCCTCCACATTCAACCTCAAACTCACC+TGGTGTTTGTGGGTGTGGTGAGTTTGAGGTTGA+GGTGAGTTTGAGGTTGAATGTGGA'
        strand_a = 'CCCTCCACATTCAACCTCAAACTCACC'  # substrate_perf_seq
        strand_b = 'TGGTGTTTGTGGGTGTGGTGAGTTTGAGGTTGA'  # incumbent_perf_seq
        strand_c = 'GGTGAGTTTGAGGTTGAATGTGGA'  # invader_perf_seq
        
    strand_list = [strand_a, strand_b, strand_c]
    ref_name_list = assign_base_names(ref_strands)
    ref_name = [item for sublist in ref_name_list for item in sublist]
    
    
    # Load data
    print(f"[Read] Loading data from {inpath}")
    
    trajs_seqs, trajs_states, trajs_times, trajs_energies = read_machinek(inpath, rxn, strand_a, num_files=224)

    # save read data
    print(f"[Read] Saving preprocessed data to {outpath}")

    data_to_save = {
    "trajs_seqs": trajs_seqs,
    "trajs_states": trajs_states,
    "trajs_times": trajs_times,
    "trajs_energies": trajs_energies,
    "ref_name": ref_name,
    "ref_name_list": ref_name_list,
    "strand_list": strand_list,
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
    

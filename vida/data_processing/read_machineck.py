import argparse
import numpy as np
import pickle
import gzip
import time
from utils import read_machinek, assign_base_names


def main():
    parser = argparse.ArgumentParser(description='Load Data')
    parser.add_argument('--inpath', required=True, help='Path to input data file')
    parser.add_argument('--rxn', required=True, help='Reaction name')
    parser.add_argument('--num-files', required=True, type=int, help='Number of files')
    parser.add_argument('--outpath', required=True, help='output file path')
    
    args = parser.parse_args()

    inpath = args.inpath
    rxn = args.rxn
    outpath = args.outpath
    num_files = args.num_files

    if rxn == "Machinek-PRF":
        strand_sub = 'CCCTCCACATTCAACCTCAAACTCACC'  # substrate (or target)
        strand_incb = 'TGGTGTTTGTGGGTGTGGTGAGTTTGAGGTTGA'  # incumbent
        strand_inv = 'GGTGAGTTTGAGGTTGAATGTGGA'  # invader
        
    if rxn == "Machinek-Mismatch2":
        strand_sub = 'CCCTCCACATTCAACCTCAAACTCACC' 
        strand_incb = 'TGGTGTTTGTGGGTGTGGTGAGTTTGAGGTTGA'  
        strand_inv = 'GGTGAGTTTGAGGTTCAATGTGGA'  
    
    if rxn == "Machinek-Mismatch10":
        strand_sub = 'CCCTCCACATACCTCAAATCACTCACC'
        strand_incb = 'TGGTGTTTGTGGGTGTGGTGAGTGATTTGAGGT'
        strand_inv = 'GGTGAGTCATTTGAGGTATGTGGA'
        
    if rxn == "Machinek-Mismatch14":
        strand_sub =  'CCCTCCACATTCAACCTCAAACTCACC'
        strand_incb = 'TGGTGTTTGTGGGTGTGGTGAGTTTGAGGTTGA'
        strand_inv = 'GGTCAGTTTGAGGTTGAATGTGGA'
        
    if rxn == "Machinek-Mismatch14C2T":
        strand_sub =  'CCCTCCACATTCAACCTCAAACTCACC'
        strand_incb = 'TGGTGTTTGTGGGTGTGGTGAGTTTGAGGTTGA'
        strand_inv = 'GGTTAGTTTGAGGTTGAATGTGGA'
    
    if rxn == "Machinek-Mismatch2C2A":
        strand_sub = 'CCCTCCACATTCAACCTCAAACTCACC' 
        strand_incb = 'TGGTGTTTGTGGGTGTGGTGAGTTTGAGGTTGA'
        strand_inv = 'GGTGAGTTTGAGGTTAAATGTGGA'  
    
    if rxn == "Machinek-Mismatch2C2T":
        strand_sub = 'CCCTCCACATTCAACCTCAAACTCACC' 
        strand_incb = 'TGGTGTTTGTGGGTGTGGTGAGTTTGAGGTTGA'
        strand_inv = 'GGTGAGTTTGAGGTTTAATGTGGA'  
                
        
    ref_strands = strand_sub + '+' + strand_incb + '+' + strand_inv
        
    strand_list = [strand_sub, strand_incb, strand_inv]
    ref_name_list = assign_base_names(ref_strands)
    ref_name = [item for sublist in ref_name_list for item in sublist]
    
    
    # Load data
    print(f"[Read] Loading data from {inpath}")
    
    trajs_seqs,trajs_states,trajs_times,trajs_energies,trajs_shortnames,trajs_incbinvpairs = read_machinek(
                                                                        inpath,
                                                                        rxn, 
                                                                        ref_name_list,
                                                                        strand_list,
                                                                        strand_sub, 
                                                                        strand_incb, 
                                                                        strand_inv, 
                                                                        num_files=num_files
                                                                        )

    # save read data
    print(f"[Read] Saving preprocessed data to {outpath}")

    data_to_save = {
    "trajs_seqs": trajs_seqs,
    "trajs_states": trajs_states,
    "trajs_times": trajs_times,
    "trajs_energies": trajs_energies,
    "trajs_shortnames": trajs_shortnames,
    "trajs_incbinvpairs": trajs_incbinvpairs,
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
    

import numpy as np
import pickle
import argparse
import time
from misc import Config, dataloader
import gzip


if __name__ == '__main__': 
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--predata', required=True, help='preprocessed data file')
    parser.add_argument('--scatter', required=True, help='scatter transform data file')
    parser.add_argument('--dist', required=True, help='distance data file')
    parser.add_argument('--fconfig', required=True, help='config file')
    parser.add_argument('--outpath', required=True, help='output dataloader')

    args = parser.parse_args()

    predata = args.predata
    scatter = args.scatter
    dist = args.dist
    fconfig = args.fconfig
    outpath = args.outpath
        
    # Load the data
    print(f"[Dataloader] Loading preprocessed data from {predata}")
    
    loaded_data = np.load(predata)
    
    energy_uniq = loaded_data["energy_uniq"]
    
    
    print(f"[Dataloader] Loading scatter transform data from {scatter}")
    
    loaded_data = np.load(scatter)
    
    scar_uniq = loaded_data["scar_uniq"]
    
    
    print(f"[Dataloader] Loading distance data from {dist}")
        
    loaded_data = np.load(dist)
        
    x_j = loaded_data["x_j"]
    d_ij = loaded_data["d_ij"]
    e_ij = loaded_data["e_ij"]
    w_ij = loaded_data["w_ij"]

    
    print(f"[Dataloader] Loading config data from {fconfig}")
    
    config = Config(fconfig)    
    
    
    # make the dataloader
    print(f"[Dataloader] Making dataloader")
    
    data_loader, train_loader, val_loader = dataloader(scar_uniq, energy_uniq, config)
    dist_loader = (w_ij, d_ij, e_ij, x_j)
    
    
    # save the dataloader in gzip format
    print(f"[Dataloader] Saving dataloader to {outpath}")
    
    data_to_save = {
    "data_loader": data_loader,
    "train_loader": train_loader,
    "val_loader": val_loader,
    "dist_loader": dist_loader,
    }
    
    
    with gzip.open(outpath, 'wb') as file:
        pickle.dump(data_to_save, file)

        
    print("[Dataloader] Done!")
    
    # Record the end time
    end_time = time.time()
    
    # Print the time elapsed
    print(f"[Dataloader] Elapsed Time: {(end_time - start_time):.3f} seconds")
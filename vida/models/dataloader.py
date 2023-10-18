import pickle
import argparse
from misc import Config, dataloader



if __name__ == '__main__': 

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
    print(f"[dataloader] Loading preprocessed data from {predata}")
    
    with open(predata, 'rb') as file:
        loaded_data = pickle.load(file)
    
    SIMS_G_uniq = loaded_data["SIMS_G_uniq"]
    
    
    print(f"[dataloader] Loading scatter transform data from {scatter}")
    
    with open(scatter, 'rb') as file:
        loaded_data = pickle.load(file)
    
    SIMS_scar_uniq = loaded_data["SIMS_scar_uniq"]
    
    
    print(f"[dataloader] Loading distance data from {dist}")
    
    with open(dist, 'rb') as file:
        loaded_data = pickle.load(file)
        
    X_j = loaded_data["X_j"]
    D_ij = loaded_data["D_ij"]
    ED_ij = loaded_data["ED_ij"]
    P_tot = loaded_data["P_tot"]

    
    print(f"[dataloader] Loading config data from {fconfig}")
    
    config = Config(fconfig)    
    
    
    # make the dataloader
    print(f"[dataloader] Making dataloader")
    
    data_loader, train_loader, val_loader = dataloader(SIMS_scar_uniq, SIMS_G_uniq, config)
    dist_loader = (P_tot, D_ij, ED_ij, X_j)
    
    
    # save the dataloader
    print(f"[dataloader] Saving dataloader to {outpath}")
    
    data_to_save = {
    "data_loader": data_loader,
    "train_loader": train_loader,
    "val_loader": val_loader,
    "dist_loader": dist_loader,
    }
    
    with open(outpath, 'wb') as file:
        pickle.dump(data_to_save, file)
    
    
    print("[dataloader] Done!")
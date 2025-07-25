import torch
import numpy as np
import gc
import pickle
import argparse
import gzip
from misc import Config, train
from vida_model import VIDA, Encoder, Decoder, Regressor


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rxn', required=True, help='Reaction name')

    args = parser.parse_args()

    reaction_id = args.rxn

    data = "data/post_data/{}/dataloader_{}.pkl.gz".format(reaction_id, reaction_id)
    fconfig = "data/post_data/{}/config_tuned.json".format(reaction_id)
    outpath = "data/post_data/{}".format(reaction_id)

    # Load the data
    print(f"[Train] Loading dataloader from {data}")
    
    with gzip.open(data, 'rb') as file:
        loaded_data = pickle.load(file)
    
    data_loader = loaded_data["data_loader"]
    train_loader = loaded_data["train_loader"]
    val_loader = loaded_data["val_loader"]
    dist_loader = loaded_data["dist_loader"]
    
    config = Config(fconfig)

    print(f"[Train] Initialize VIDA model")
    
    ## Retrain the model if Nan loss occurs ##
    retry = 0
    max_retries = 3
    
    while retry < max_retries:
        # Get the input dimension
        input_dim = data_loader.dataset[0][0].shape[0]
        
        encoder = Encoder(input_dim=input_dim, hidden_dim=config.hidden_dim, latent_dim=config.latent_dim)
        decoder = Decoder(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim, output_dim=input_dim)
        regressor = Regressor(latent_dim=config.latent_dim)
        
        # Initialize ViDa 
        vida = VIDA(encoder, decoder, regressor)
        
        # Define optimizer
        optimizer = torch.optim.Adam(vida.parameters(), lr=config.learning_rate)
        
        # Define scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        
        print (f"[Train] Start training VIDA model")
        
        # Train VIDA
        val_loss, *_  = train(fconfig, vida, data_loader, train_loader, val_loader, dist_loader, optimizer, scheduler, outpath, neigh_mode='repeat')
        
        if np.isnan(val_loss):
            print(f"[Train] NaN Loss encountered. Restarting training...")
            retry += 1
            vida = None  # clear the model to free memory
            gc.collect() # free up cpu
            torch.cuda.empty_cache()  # clear gpu cache
            continue # restart training
        else:
            print (f"[Train] Saving VIDA model to {outpath}")
            print (f"[Train] Training DONE!")
            break # training successful
            

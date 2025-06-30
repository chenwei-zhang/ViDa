import torch
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

    data = "vida/data/post_data/{}/dataloader_{}.pkl.gz".format(reaction_id, reaction_id)
    fconfig = "vida/data/post_data/{}/config_template.json".format(reaction_id)
    outpath = "vida/data/post_data/{}".format(reaction_id)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.lr_patience, verbose=True)
    
    print (f"[Train] Start training VIDA model")
    
    # Train VIDA
    ## neigh_mode='unique' or 'repeat'
    train(fconfig, vida, data_loader, train_loader, val_loader, dist_loader, optimizer, scheduler, outpath, neigh_mode='repeat')
    
    
    print (f"[Train] Saving VIDA model to {outpath}")
    
    print (f"[Train] Training DONE!")

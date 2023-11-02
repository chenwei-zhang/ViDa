import torch
import pickle
import argparse
import gzip
from misc_cuda import Config, train
from vida_model_cuda import VIDA, Encoder, Decoder, Regressor


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='load dataloader')
    parser.add_argument('--fconfig', required=True, help='config file')
    parser.add_argument('--outpath', required=True, help='output dataloader')

    args = parser.parse_args()

    data = args.data
    fconfig = args.fconfig
    outpath = args.outpath
        
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
    train(fconfig, vida, data_loader, train_loader, val_loader, dist_loader, optimizer, scheduler, outpath)
    
    
    print (f"[Train] Saving VIDA model to {outpath}")
    
    print (f"[Train] Training DONE!")

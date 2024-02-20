import torch
import pickle
import argparse
import gzip
from misc_cuda import Config, train
from vida_model_cuda import VIDA, Encoder, Decoder, Regressor
import optuna
import json
import warnings
warnings.filterwarnings("ignore")



def objective(trial):

    with gzip.open(data, 'rb') as file:
        loaded_data = pickle.load(file)
    
    data_loader = loaded_data["data_loader"]
    train_loader = loaded_data["train_loader"]
    val_loader = loaded_data["val_loader"]
    dist_loader = loaded_data["dist_loader"]
    
    config = Config(fconfig)

    # Suggest hyperparameters
    config.alpha = trial.suggest_float('alpha', 0.1, 10)
    config.beta = trial.suggest_float('beta', 1e-5, 1e-3)
    config.gamma = trial.suggest_float('gamma', 0.1, 1)
    config.delta = trial.suggest_float('delta', 0.01, 0.1)
    config.epsilon = trial.suggest_float('epsilon', 1e-6, 1e-4)

    # update hyperparameters:
    config.update({'alpha': config.alpha})
    config.update({'beta': config.beta})
    config.update({'gamma': config.gamma})
    config.update({'delta': config.delta})
    config.update({'epsilon': config.epsilon})

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

    # Train VIDA with the suggested hyperparameters
    val_loss_final = train(fconfig, vida, data_loader, train_loader, val_loader, dist_loader, optimizer, scheduler, outpath, neigh_mode='repeat')
    
    return val_loss_final


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='load dataloader')
    parser.add_argument('--fconfig', required=True, help='config file')
    parser.add_argument('--outpath', required=True, help='output dataloader')

    args = parser.parse_args()

    data = args.data
    fconfig = args.fconfig
    outpath = args.outpath

    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)  # Adjust the number of trials based on your computational resources
    
    best_hyperparams = study.best_params
    print("Best hyperparameters:", best_hyperparams)

    # Specify the file path where you want to save the hyperparameters
    output_file_path = 'best_hyperparameters.json'

    # Write the best hyperparameters to a JSON file
    with open(output_file_path, 'w') as outfile:
        json.dump(best_hyperparams, outfile, indent=4)

    print(f"Best hyperparameters saved to {output_file_path}")
    
    # # Load the data
    # print(f"[Train] Loading dataloader from {data}")
    # with gzip.open(data, 'rb') as file:
    #     loaded_data = pickle.load(file)
    # data_loader = loaded_data["data_loader"]
    # train_loader = loaded_data["train_loader"]
    # val_loader = loaded_data["val_loader"]
    # dist_loader = loaded_data["dist_loader"]
    # config = Config(fconfig)

    # print(f"[Train] Initialize VIDA model")
    # # Get the input dimension
    # input_dim = data_loader.dataset[0][0].shape[0]
    # encoder = Encoder(input_dim=input_dim, hidden_dim=config.hidden_dim, latent_dim=config.latent_dim)
    # decoder = Decoder(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim, output_dim=input_dim)
    # regressor = Regressor(latent_dim=config.latent_dim)
    # # Initialize ViDa 
    # vida = VIDA(encoder, decoder, regressor)
    # # Define optimizer
    # optimizer = torch.optim.Adam(vida.parameters(), lr=config.learning_rate)
    # # Define scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.lr_patience, verbose=True)
    # print (f"[Train] Start training VIDA model")
    # # Train VIDA
    # ## neigh_mode='unique' or 'repeat'
    # train(fconfig, vida, data_loader, train_loader, val_loader, dist_loader, optimizer, scheduler, outpath, neigh_mode='repeat')    
    # print (f"[Train] Saving VIDA model to {outpath}")
    # print (f"[Train] Training DONE!")

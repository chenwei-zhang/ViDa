import os
import argparse
import tempfile
import shutil
import json
import gzip
import pickle
import torch
import optuna
from misc import Config, train
from vida_model import VIDA, Encoder, Decoder, Regressor



# ---------------------------
# Helpers
# ---------------------------
def load_data(reaction_id):
    data_path = f"data/post_data/{reaction_id}/dataloader_{reaction_id}.pkl.gz"
    with gzip.open(data_path, 'rb') as f:
        loaded = pickle.load(f)
    return loaded["data_loader"], loaded["train_loader"], loaded["val_loader"], loaded["dist_loader"]

def build_model(config, input_dim):
    encoder = Encoder(input_dim=input_dim, hidden_dim=config.hidden_dim, latent_dim=config.latent_dim)
    decoder = Decoder(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim, output_dim=input_dim)
    regressor = Regressor(latent_dim=config.latent_dim)
    return VIDA(encoder, decoder, regressor)



# ---------------------------
# Optuna objective
# ---------------------------
def make_objective(reaction_id, base_config):
    def objective(trial):
        # Suggest hyperparameters
        lr      = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        alpha   = trial.suggest_float('alpha', 0.1, 5.0) # linear scaling
        beta    = trial.suggest_float('beta', 1e-6, 1e-2, log=True)
        gamma   = trial.suggest_float('gamma', 0.1, 1.0) # linear scaling
        delta   = trial.suggest_float('delta', 1e-5, 1e-2, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-6, 1e-3, log=True)

        # Make a temporary config file for this trial
        tmp_config = tempfile.mktemp(suffix=".json")
        shutil.copy(base_config, tmp_config)
        cfg = Config(tmp_config)
        cfg.update({
            "reaction_id": reaction_id,
            "learning_rate": lr,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "epsilon": epsilon,
            "n_epochs": 10 # shorten epochs to speed up tuning
        })

        # Load data and build model
        data_loader, train_loader, val_loader, dist_loader = load_data(reaction_id)
        input_dim = data_loader.dataset[0][0].shape[0]
        model = build_model(cfg, input_dim)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

        # Train and get validation loss
        val_loss, *_ = train(tmp_config, model, data_loader, train_loader, val_loader,
                             dist_loader, optimizer, scheduler,
                             outpath=f"data/post_data/{reaction_id}",
                             neigh_mode='repeat')
        return val_loss
    
    return objective



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rxn', required=True, help='Reaction name (folder)')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    args = parser.parse_args()

    reaction_id = args.rxn
    
    # Config paths
    src_config = "vida/models/config_template.json"
    base_config = f"data/post_data/{reaction_id}/config_tuned.json"
    shutil.copy(src_config, base_config)
    print(f"[Tuning] Copied {src_config} to {base_config}")

    with open(base_config, 'r') as f:
        config_data = json.load(f)
    max_epochs = config_data["n_epochs"]
    
    print(f"[Tuning] Starting Optuna tuning for reaction: {reaction_id}")
    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(reaction_id, base_config), n_trials=args.trials)

    print("[Tuning] Done! Best parameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    # Update the config template with best params
    best_params = study.best_params
    best_params["n_epochs"] = max_epochs # Set to full epochs for final training
    config_data.update(best_params)
    with open(base_config, 'w') as f:
        json.dump(config_data, f, indent=4)

    print(f"[Tuning] Updated {base_config} with best hyperparameters.")



if __name__ == "__main__":
    main()

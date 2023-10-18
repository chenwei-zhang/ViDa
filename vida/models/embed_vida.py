import torch
import pickle
import argparse
from misc import Config
from vida_model import VIDA, Encoder, Decoder, Regressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import phate



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='load dataloader')
    parser.add_argument('--model', required=True, help='tranined model.pt file')
    parser.add_argument('--fconfig', required=True, help='config file in the model path')
    parser.add_argument('--outpath', required=True, help='output embedding data')

    args = parser.parse_args()

    data = args.data
    model = args.model
    outpath = args.outpath
    fconfig = args.fconfig
    
    # Load the data
    print(f"[Embed] Loading dataloader from {data}")
    
    with open(data, 'rb') as file:
        loaded_data = pickle.load(file)
    
    data_loader = loaded_data["data_loader"]
    
    config = Config(fconfig)
    
    # Load the trained model
    print(f"[Embed] Load trained VIDA model")
    
    input_dim = data_loader.dataset[0][0].shape[0]
    
    encoder = Encoder(input_dim=input_dim, hidden_dim=config.hidden_dim, latent_dim=config.latent_dim)
    decoder = Decoder(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim, output_dim=input_dim)
    regressor = Regressor(latent_dim=config.latent_dim)
    
    # Initialize ViDa 
    vida = VIDA(encoder, decoder, regressor)
    # Load the trained parameters
    vida.load_state_dict(torch.load(model))
    
    
    # Do the embedding
    print(f"[Embed] Embedding data")
    
    vida.to(config.device).eval()
    
    embeddings = vida.get_embeddings(data_loader.dataset.tensors[0].to(config.device))
    
    # Put the embeddings to cpu and convert to numpy array
    with torch.no_grad():
        embeddings = embeddings.to('cpu').numpy()
    
    print(f'[Embed] Embedding maximum: {embeddings.max()}, minimum: {embeddings.min()}, mean: {embeddings.mean()}, std: {embeddings.std()}') 
    
    
    # Do PCA (n_components=2)
    print(f"[Embed] Do PCA")
    
    pca_coords = PCA(n_components=2).fit_transform(embeddings)
    
    print(f"[Embed] PCA maximum: {pca_coords.max()}, minimum: {pca_coords.min()}, mean: {pca_coords.mean()}, std: {pca_coords.std()}")
    
    
    # Do PHATE (n_components=2)
    print(f"[Embed] Do PHATE")
    
    scaler = StandardScaler()
    data_embed = scaler.fit_transform(embeddings)
    
    phate_operator = phate.PHATE(n_jobs=-2)
    phate_coords = phate_operator.fit_transform(data_embed)
    
    print(f"[Embed] PHATE maximum: {phate_coords.max()}, minimum: {phate_coords.min()}, mean: {phate_coords.mean()}, std: {phate_coords.std()}")
    
    
    # Save the embedding data
    print(f"[Embed] Saving embedding data to {outpath}")
    
    data_to_save = {
    "embeddings": embeddings,
    "pca_coords": pca_coords,
    "phate_coords": phate_coords
    }
    
    with open(outpath, 'wb') as file:
        pickle.dump(data_to_save, file)
        
    print(f"[Embed] Embedding DONE!")    
        
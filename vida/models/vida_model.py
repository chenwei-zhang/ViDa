import torch
import torch.nn as nn
import torch.nn.functional as F


# Encoder
class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        '''
        Args:
        ----
            - input_dim: the dimension of the input node feature
            - hiddent_dim: the dimension of the hidden layer
            - latent_dim: the dimension of the latent space (bottleneck layer)
        '''
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(400, 400)
        self.bn2 = nn.BatchNorm1d(400)
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution, note how we only output
        # diagonal values of covariance matrix. Here we assume
        # they are conditionally independent
        self.hid2mu = nn.Linear(400, self.latent_dim)
        self.hid2logvar = nn.Linear(400, self.latent_dim)
        
    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        mu = self.hid2mu(x)
        logvar = self.hid2logvar(x)
        return mu, logvar


# Decoder
class Decoder(nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, output_dim):
        '''
        Args:
        ----
            - latent_dim: the dimension of the latent space (bottleneck layer)
            - hiddent_dim: the dimension of the hidden layer
            - output_dim: the dimension of the output node feature
        '''
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.latent_dim, 400)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc2 = nn.Linear(400, 400)
        self.bn2 = nn.BatchNorm1d(400)
        self.fc3 = nn.Linear(400, self.output_dim)
        
    def forward(self, z):
        x = self.bn1(F.relu(self.fc1(z)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    
# Regressor
class Regressor(nn.Module):
    
    def __init__(self, latent_dim):
        '''
        The regressor is used to predict the energy of the node
        
        Args:
        ----
            - latent_dim: the dimension of the latent space (bottleneck layer)
        '''
        super(Regressor, self).__init__()
        self.latent_dim = latent_dim
        
        self.regfc1 = nn.Linear(self.latent_dim, 15)
        self.regfc2 = nn.Linear(15, 1)
        
    def forward(self, z):
        y = F.relu(self.regfc1(z))
        y = self.regfc2(y)
        return y
    

# ViDa model
class VIDA(nn.Module):
    
    def __init__(self, encoder, decoder, regressor):
        '''
        Args:
        ----
            - input_dim: the dimension of the input node feature
            - hiddent_dim: the dimension of the hidden layer
            - latent_dim: the dimension of the latent space (bottleneck layer)
            - output_dim: the dimension of the output node feature (same as input_dim)
        '''
        super(VIDA, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.regressor = regressor
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        y_pred = self.regressor(z)
        return x_recon, y_pred, z, mu, logvar
    
    def get_embeddings(self, x):
        """
        Perform inference and get the latent embeddings.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Latent embeddings.
        """
        mu, _ = self.encoder(x)
        return mu
    
    # VAE loss
    def vae_loss(self, x_recon, x, mu, logvar):
        '''
        Compute the VAE loss
        
        Args:
            - x_recon: the reconstructed node feature
            - x: the original node feature
            - mu: the mean of the latent space
            - logvar: the log variance of the latent space
        
        Returns:
        - loss: PyTorch Tensor containing (scalar) the loss for the VAE
        '''
        recon_loss = F.mse_loss(x_recon.flatten(), x.flatten()) # L2 loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_loss
    
    # energy prediction loss
    def pred_loss(self, y_pred, y):
        '''
        Compute the energy prediction loss
        
        Args:
        ----
            - y_pred: the predicted energy of the node
            - y: the true energy of the node
        
        Returns:
            - loss: PyTorch Tensor containing (scalar) the loss for the prediction
        '''
        return F.mse_loss(y_pred.flatten(), y.flatten())
    
    # minimum passage time loss
    def mpt_loss(self, device, zi, zj, D_ij, P_tot, idx, batchXj_id):
        '''
        Compute the distance loss between embeddings 
        and the minimum expected holding time
        
        Args:
            - device: the device (cpu or cuda or mps) to run the model
            - zi: the embedding of the node i
            - zj: the embedding of the nodes j's
            - D_ij: the post-processing distance between nodes i and j's
            - P_tot: the total probability of the nodes i and j's
            - idx: the index of the node i
            - batchXj_id: the index of the nodes j's
        
        Returns:
        - loss: PyTorch Tensor containing (scalar) the mpt loss for the embedding distance
        '''
        zi = zi.reshape(-1,1,zi.shape[-1])
        l2_zizj = torch.sqrt(torch.sum((zi-zj)**2, dim=-1))
        dist_diff = (l2_zizj - (D_ij[idx]).to(device))**2
        wij = (P_tot[idx].reshape(-1,1) * P_tot[batchXj_id]).to(device)
        mpt_loss = torch.sum(dist_diff * wij)
        return mpt_loss
    
    # graph edit distance loss
    def ged_loss(self, device, zi, zj, ED_ij, idx):
        '''
        Compute the edit distance loss between embeddings 
        
        Args:
            - device: the device (cpu or cuda or mps) to run the model
            - zi: the embedding of the node i
            - zj: the embedding of the nodes j's
            - ED_ij: the post-processing distance between nodes i and j's
            - idx: the index of the node i
            - batchXj_id: the index of the nodes j's
        
        Returns:
        - loss: PyTorch Tensor containing (scalar) the ged loss for the embedding distance
        '''
        zi = zi.reshape(-1,1,zi.shape[-1])
        l2_zizj = torch.sqrt(torch.sum((zi-zj)**2, dim=-1))
        dist_diff = (l2_zizj - (ED_ij[idx]).to(device))**2
        ged_loss = torch.sum(dist_diff)
        return ged_loss
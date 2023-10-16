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
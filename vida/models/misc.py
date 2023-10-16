import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# make data loader
def dataloader(SIMS_scar_uniq, SIMS_G_uniq, config):
    data_tup = (torch.Tensor(SIMS_scar_uniq),
                torch.Tensor(SIMS_G_uniq),
                torch.arange(len(SIMS_scar_uniq)))
    data_dataset = torch.utils.data.TensorDataset(*data_tup)

    # split the dataset into train and validation
    data_size = len(data_dataset)
    train_size = int(0.7 * data_size)
    val_size = data_size - train_size

    train_data, val_data = torch.utils.data.random_split(data_dataset, [train_size, val_size], 
                                                         generator=torch.Generator().manual_seed(42))

    print(data_size, len(train_data), len(val_data))

    data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=config.batch_size,
                                              shuffle=False, num_workers=0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, 
                                               shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size,
                                             shuffle=False, num_workers=0)

    return data_loader, train_loader, val_loader


## loss functions
# VAE loss
def vae_loss(x_recon, x, mu, logvar):
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
    BCE = F.mse_loss(x_recon.flatten(), x.flatten()) # L2 loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD


# prediction loss
def pred_loss(y_pred, y):
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
def mpt_loss(config, zi, zj, Dij, P_tot, idx, batchXj_id):
    '''
    Compute the mpt loss between embeddings 
    and the minimum expected holding time
    
    Args:
        - zi: the embedding of the node i
        - zj: the embedding of the nodes j's
        - Dij: the post-processing distance between nodes i and j's
        - P_tot: the total probability of the nodes i and j's
        - idx: the index of the node i
        - batchXj_id: the index of the nodes j's
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the embedding distance
    '''
    zi = zi.reshape(-1,1,zi.shape[-1])
    l2_zizj = torch.sqrt(torch.sum((zi-zj)**2, dim=-1))
    dist_diff = (l2_zizj - (Dij[idx]).to(config.device))**2

    wij = (P_tot[idx].reshape(-1,1) * P_tot[batchXj_id]).to(config.device) # importance weight of nodes i and j       
    dist_loss = torch.sum(dist_diff * wij)
    return dist_loss


# graph edit distance loss
def ged_loss(config, zi, zj, ED_ij, idx):
    '''
    Compute the edit distance loss between embeddings 
    
    Args:
        - zi: the embedding of the node i
        - zj: the embedding of the nodes j's
        - ED_ij: the post-processing distance between nodes i and j's
        - idx: the index of the node i
        - batchXj_id: the index of the nodes j's
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the embedding distance
    '''
    zi = zi.reshape(-1,1,zi.shape[-1])
    l2_zizj = torch.sqrt(torch.sum((zi-zj)**2, dim=-1))
    dist_diff = (l2_zizj - (ED_ij[idx]).to(config.device))**2
    editdist_loss = torch.sum(dist_diff)
    return editdist_loss
    
    
# early stopping function
def early_stop(val_loss, epoch, patience):
    """
    Check if validation loss has not improved for a certain number of epochs
    
    Args:
        val_loss (float): current validation loss
        epoch (int): current epoch number
        patience (int): number of epochs to wait before stopping if validation loss does not improve
        
    Returns:
        bool: True if validation loss has not improved for the last `patience` epochs, False otherwise
    """
    if epoch == 0:
        # First epoch, don't stop yet
        return False
    else:
        # Check if validation loss has not improved for `patience` epochs
        if val_loss >= early_stop.best_loss:
            early_stop.num_epochs_without_improvement += 1
            if early_stop.num_epochs_without_improvement >= patience:
                print("Stopping early")
                return True
            else:
                return False
        else:
            # Validation loss improved, reset counter
            early_stop.best_loss = val_loss
            early_stop.num_epochs_without_improvement = 0
            return False
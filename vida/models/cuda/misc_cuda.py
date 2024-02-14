import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import time
import os
import shutil
import gc

# load and update the original config file
class Config:
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, 'r') as config_file:
            self.config = json.load(config_file)

    def __getattr__(self, attr):
        return self.config.get(attr, None)

    def update(self, new_values):
        self.config.update(new_values)
        with open(self.file_path, 'w') as config_file:
            json.dump(self.config, config_file, indent=4)



# make data loader
def dataloader(scar_uniq, energy_uniq, config, ratio=0.7):
    data_tup = (torch.Tensor(scar_uniq),
                torch.Tensor(energy_uniq),
                torch.arange(len(scar_uniq)))
    data_dataset = torch.utils.data.TensorDataset(*data_tup)

    # split the dataset into train and validation
    data_size = len(data_dataset)
    train_size = int(ratio * data_size)
    val_size = data_size - train_size

    train_data, val_data = torch.utils.data.random_split(data_dataset, [train_size, val_size], 
                                                         generator=torch.Generator().manual_seed(42))

    data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=config.batch_size,
                                              shuffle=False, num_workers=0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, 
                                               shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size,
                                             shuffle=False, num_workers=0)

    return data_loader, train_loader, val_loader

    
    
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
        

#  Get mpt,ged loss embedding neighbors for training loss
def embed_neighbors(neigh_mode, device, model, data_loader, input_dim, batch_xj_id, batch_xdj_id, batch_xej_id, knn_mpt, knn_ged):

    if neigh_mode == 'unique':
        ## embed the unique neighbors ##
        batch_uniq_id = torch.unique(batch_xj_id.flatten())
        neighbor_input = data_loader.dataset.tensors[0].to(device)[batch_uniq_id].reshape(-1, input_dim)
        _, _, neighbor_embed, _, _ = model(neighbor_input)     # with noise
        # neighbor_embed = model.get_embeddings(neighbor_input)    # without noise
        '''make a dictionary for mpt and ged neighbors search 
            [Note: put to cpu due to mps incompatible with cuda] 
        '''              
        my_dict = {key.item(): val for key, val in zip(batch_uniq_id, neighbor_embed)}
        # mpt embedding
        neighbor_mpt = [torch.unsqueeze(my_dict[key], 1) for key in batch_xdj_id.flatten().numpy()]
        neighbor_mpt = torch.cat(neighbor_mpt, dim=0).reshape(-1, knn_mpt, neighbor_embed.shape[-1])
        # ged embedding
        neighbor_ged = [torch.unsqueeze(my_dict[key], 1) for key in batch_xej_id.flatten().numpy()]
        neighbor_ged = torch.cat(neighbor_ged, dim=0).view(-1, knn_ged, neighbor_embed.shape[-1])
    
    elif neigh_mode == 'repeat':
        ## embed all repeated neighbors ##
        # mpt embedding
        neighbor_input_mpt = data_loader.dataset.tensors[0].to(device)[batch_xdj_id].reshape(-1, input_dim)
        _, _, neighbor_mpt, _, _ = model(neighbor_input_mpt)
        neighbor_mpt = neighbor_mpt.reshape(-1, knn_mpt, neighbor_mpt.shape[-1])
        # ged embedding
        neighbor_input_ged = data_loader.dataset.tensors[0].to(device)[batch_xej_id].reshape(-1, input_dim)
        _, _, neighbor_ged, _, _ = model(neighbor_input_ged)
        neighbor_ged = neighbor_ged.reshape(-1, knn_ged, neighbor_ged.shape[-1])
                               
    return neighbor_mpt, neighbor_ged



# validate vida
def validate(config, model, data_loader, val_loader, p_i, d_ij, e_ij, x_dj, x_ej, x_j, neigh_mode):
    
    # Get the input dimension
    input_dim = data_loader.dataset[0][0].shape[0]
    
    model.to(config.device)
    model.eval()
    
    val_loss = 0; val_bce = 0; val_kld = 0; val_pred = 0; val_mpt = 0; val_ged = 0
    
    # Disable gradient calculation to speed up inference
    with torch.no_grad():
        for x, y, idx in val_loader:
            
            # Configure input
            x = x.to(config.device)
            y = y.to(config.device)
            idx = idx.to(config.device)
            
            knn_mpt = x_dj.shape[-1]
            knn_ged = x_ej.shape[-1]
            
            # forward x_j
            model.eval()
            with torch.no_grad():
                batch_xj_id = x_j[idx]
                batch_xdj_id = x_dj[idx]
                batch_xej_id = x_ej[idx]
                
                neighbor_mpt, neighbor_ged = embed_neighbors(neigh_mode, config.device, model, data_loader, input_dim, batch_xj_id, batch_xdj_id, batch_xej_id, knn_mpt, knn_ged)
                
            
            # embedding
            x_recon, y_pred, z, mu, logvar = model(x)
            
            # # compute the total loss
            # vae loss
            recon_loss, kl_loss = model.vae_loss(x_recon, x, mu, logvar)
            recon_loss = recon_loss.item()
            kl_loss = kl_loss.item()
            
            # energy prediction loss
            p_loss = model.pred_loss(y_pred, y).item()
                
            # distance loss            
            t_loss = model.mpt_loss(z, neighbor_mpt, d_ij, p_i, idx, x_dj[idx]).item()
            
            # edit distance loss
            e_loss = model.ged_loss(z, neighbor_ged, e_ij, idx).item()
            
            # scaling the loss
            recon_loss = config.alpha * recon_loss
            kl_loss = config.beta * kl_loss
            p_loss = config.gamma * p_loss
            t_loss = config.delta * t_loss
            e_loss = config.epsilon * e_loss
            
            # total loss
            loss = recon_loss + kl_loss + p_loss + t_loss + e_loss
            
            val_loss += loss
            val_bce += recon_loss
            val_kld += kl_loss
            val_pred += p_loss
            val_mpt += t_loss
            val_ged += e_loss
                        
    print('Validation Loss: {:.4f}'.format(val_loss/len(val_loader.dataset)))
    
    # Clear the cache
    gc.collect()
    # Clear the cache
    torch.cuda.empty_cache()
    # torch.mps.empty_cache()
    
    return val_loss/len(val_loader.dataset), val_bce/len(val_loader.dataset), val_kld/len(val_loader.dataset), val_pred/len(val_loader.dataset), val_mpt/len(val_loader.dataset), val_ged/len(val_loader.dataset)
            

       
# train vida     
def train(fconfig, model, data_loader, train_loader, val_loader, dist_loader, optimizer, scheduler, outpath, neigh_mode='unique'):
    
    '''
    Train VIDA!
    
    Args:
    ----
        - fconfig: Experiment configurations file
        - model: Pytorch VIDA model
        - data_loader: Pytorch DataLoader for all data
        - train_loader: Pytorch DataLoader for training set
        - val_loader: Pytorch DataLoader for validation set
        - dist_loader: the tuple of (p_i, d_ij, e_ij, x_dj, x_ej)
            - p_i: the total probability of each node
            - d_ij: the shortest path distance between node i and j
            - e_ij: the graph edit distance between node i and j
            - x_dj: the index of k nearest neighbours of each node based on mpt
            - x_ej: the index of k nearest neighbours of each node based on ged
        - optimizer: Pytorch optimizer
        - scheduler: Pytorch learning rate scheduler
        - outpath: output directory
        - neigh_mode: the mode of computing the mpt and ged embedding neighbors
        - early_stop: early stopping object
    '''
    
    # create the log directory
    if not os.path.exists(f'{outpath}/model_config'):
        os.makedirs(f'{outpath}/model_config')
    
    log_time = time.strftime("%y-%m%d-%H%M")
    log_dir = f'{outpath}/model_config/{log_time}'
    
    # write the log to tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    
    # copy the config file to the output directory
    shutil.copy(fconfig, f'{log_dir}/config.json')
    
    # load the new config file
    config = Config(f'{log_dir}/config.json')
    
    # write the log time to the config file
    config.update({'log_time': log_time})
    
    # Get the input dimension
    input_dim = data_loader.dataset[0][0].shape[0]
    
    # move the model to the device (GPU or CPU)
    model.to(config.device)


    # Initialize early stop object
    early_stop.best_loss = np.Inf
    early_stop.num_epochs_without_improvement = 0

    # convert the dist_loader numpy array to tensor
    p_i, d_ij, e_ij, x_dj, x_ej = dist_loader.values()
    x_j = np.concatenate((x_dj, x_ej), axis=1)
    
    p_i = torch.from_numpy(p_i.astype(np.float32))
    d_ij = torch.from_numpy(d_ij.astype(np.float32))
    e_ij = torch.from_numpy(e_ij.astype(int))
    x_dj = torch.from_numpy(x_dj.astype(int))
    x_ej = torch.from_numpy(x_ej.astype(int))
    x_j = torch.from_numpy(x_j.astype(int))
    
    p_i = p_i.to(config.device)
    d_ij = d_ij.to(config.device)
    e_ij = e_ij.to(config.device)
    x_dj = x_dj.to(config.device)
    x_ej = x_ej.to(config.device)
    x_j = x_j.to(config.device)
    
    knn_mpt = x_dj.shape[-1]
    knn_ged = x_ej.shape[-1]
    
    config.update({'knn_mpt': knn_mpt})
    config.update({'knn_ged': knn_ged})
    config.update({'split_ratio': round(len(train_loader.dataset.indices) / data_loader.dataset.tensors[0].shape[0], 2)})
    config.update({'neigh_mode': neigh_mode})
    
        
    print('\n ------- Start Training -------')
    for epoch in range(config.n_epochs):
        start_time = time.time()
        training_loss = 0
        
        for batch_idx, (x, y, idx) in enumerate(train_loader):  # mini batch
            
            # Configure input
            x = x.to(config.device)
            y = y.to(config.device)
            idx = idx.to(config.device)
            
            # ------------------------------------------
            #  Get mpt,ged loss embedding neighbors
            # ------------------------------------------
            model.eval()
            with torch.no_grad():
                batch_xj_id = x_j[idx]
                batch_xdj_id = x_dj[idx]
                batch_xej_id = x_ej[idx]
                
                neighbor_mpt, neighbor_ged = embed_neighbors(neigh_mode, config.device, model, data_loader, input_dim, batch_xj_id, batch_xdj_id, batch_xej_id, knn_mpt, knn_ged)
                
            # ------------------------------------------
            #  Train VIDA
            # ------------------------------------------
            model.train()
            optimizer.zero_grad()
            
            # get the reconstructed nodes, predicted energy, and the embeddings
            x_recon, y_pred, z, mu, logvar = model(x)
        
            ## compute the total loss ##
            # vae loss
            recon_loss, kl_loss = model.vae_loss(x_recon, x, mu, logvar)
            
            # energy prediction loss
            p_loss = model.pred_loss(y_pred, y)
            
            # distance loss
            t_loss = model.mpt_loss(z, neighbor_mpt, d_ij, p_i, idx, batch_xdj_id)
            
            # edit distance loss
            e_loss = model.ged_loss(z, neighbor_ged, e_ij, idx)
            
            # scaling the loss
            recon_loss = config.alpha * recon_loss
            kl_loss = config.beta * kl_loss
            p_loss = config.gamma * p_loss
            t_loss = config.delta * t_loss
            e_loss = config.epsilon * e_loss
            
            # total loss
            loss = recon_loss + kl_loss + p_loss + t_loss + e_loss
            
            # backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            
            # ------------------------------------------
            # Log Progress
            # ------------------------------------------
            if batch_idx % config.log_interval == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'.format(
                    epoch, config.n_epochs-1, batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))
                
                writer.add_scalar('train_recon loss',
                                  recon_loss.item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train_kl loss',
                                  kl_loss.item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train_pred loss',
                                  p_loss.item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train_mpt loss',
                                  t_loss.item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train_ged loss',
                                  e_loss.item(),
                                  epoch * len(train_loader) + batch_idx)
    
        print ('====> Epoch: {} Average loss: {:.4f}'.format(epoch, training_loss/len(train_loader.dataset)))
        writer.add_scalar('training loss', training_loss/len(train_loader.dataset), epoch)
                
        # ------------------------------------------
        #  Validate VIDA
        # ------------------------------------------
        val_loss, val_bce, val_kld, val_pred, val_mpt, val_ged = validate(config, model, data_loader, val_loader, p_i, d_ij, e_ij, x_dj, x_ej, x_j, neigh_mode)
        
        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('val_recon loss', val_bce, epoch)
        writer.add_scalar('val_kl loss', val_kld, epoch)
        writer.add_scalar('val_pred loss', val_pred, epoch)
        writer.add_scalar('val_mpt loss', val_mpt, epoch)
        writer.add_scalar('val_ged loss', val_ged, epoch)
        
        # log the learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning rate', current_lr, epoch)
        
        # update the learning rate
        scheduler.step(val_loss)
        
        # timing
        end_time = time.time()
        epoch_time = end_time - start_time
        print (f'Epoch {epoch} train+val time: {epoch_time:.2f} seconds \n')
        
        # save the model checkpoint every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'{log_dir}/checkpoint_epoch_{epoch}.pt')           
        
        # Check if validation loss has not improved for `patience` epochs
        if early_stop(val_loss, epoch, config.patience):
            break
        
        # Clear the cache
        gc.collect()
        # torch.mps.empty_cache()  ## not available yet
        torch.cuda.empty_cache()
       
    config.update({'Training finished at epoch': epoch})
    
    writer.close()
    print('\n ------- Finished Training -------')
    
    # save the model
    torch.save(model.state_dict(), f'{log_dir}/model.pt')

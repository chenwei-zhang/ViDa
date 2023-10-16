import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter



# validate vida
def validate(config, model, data_loader, val_loader, P_tot, Dij, ED_ij, X_j,
             vae_loss, pred_loss, distance_knn_loss, edit_distance_loss):
    model.to(config.device)
    model.eval()
    
    val_loss = 0; val_bce = 0; val_kld = 0; val_pred = 0; val_dist = 0; val_edit = 0
    
    # Disable gradient calculation to speed up inference
    with torch.no_grad():
        for x, y, idx in val_loader:
            
            # Configure input
            x = x.to(config.device)
            y = y.to(config.device)
            
            # forward X_j
            model.eval()
            with torch.no_grad():
                batchXj_id = X_j[idx]
                neighbor_input = data_loader.dataset.tensors[0][batchXj_id].reshape(-1, config.input_dim).to(config.device)
                _, _, neighbor_embed, _, _ = model(neighbor_input)
                neighbor_embed = neighbor_embed.reshape(-1, config.knn, neighbor_embed.shape[-1])
                
            # embedding
            x_recon, y_pred, z, mu, logvar = model(x)
            
            # # compute the total loss
            # vae loss
            recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
            recon_loss = recon_loss.item()
            kl_loss = kl_loss.item()
            
            # energy prediction loss
            p_loss = pred_loss(y_pred, y).item()
                
            # distance loss
            dist_loss = distance_knn_loss(config, z, neighbor_embed, Dij, P_tot, idx, batchXj_id).item()
            
            # edit distance loss
            edit_loss = edit_distance_loss(config, z, neighbor_embed, ED_ij, idx).item()
            
            # scaling the loss
            recon_loss = config.alpha * recon_loss
            kl_loss = config.beta * kl_loss
            p_loss = config.gamma * p_loss
            dist_loss = config.delta * dist_loss
            edit_loss = config.epsilon * edit_loss
            
            # total loss
            loss = recon_loss + kl_loss + p_loss + dist_loss + edit_loss
            
            val_loss += loss
            val_bce += recon_loss
            val_kld += kl_loss
            val_pred += p_loss
            val_dist += dist_loss
            val_edit += edit_loss
                        
    print('Validation Loss: {:.4f}'.format(val_loss/len(val_loader.dataset)))
    
    # Clear the cache
    torch.cuda.empty_cache()
    # torch.mps.empty_cache()
    
    return val_loss/len(val_loader.dataset), val_bce/len(val_loader.dataset), val_kld/len(val_loader.dataset), val_pred/len(val_loader.dataset), val_dist/len(val_loader.dataset), val_edit/len(val_loader.dataset)
            
            
# train vida     
def train(config, model, data_loader, train_loader, val_loader, P_tot, Dij, ED_ij, X_j,
          optimizer, vae_loss, pred_loss, distance_knn_loss, edit_distance_loss, early_stop,):
    '''
    Train VIDA!
    
    Args:
    ----
        - config: Experiment configurations
        - model: Pytorch VIDA model
        - data_loader: Pytorch DataLoader for all data
        - train_loader: Pytorch DataLoader for training set
        - val_loader: Pytorch DataLoader for validation set
        - P_tot: the total probability of each node
        - Dij: the shortest path distance between k nearest pairs of nodes
        - X_j: the index of k nearest neighbours of each node
        - optimizer: Pytorch optimizer
        - vae_loss: the VAE loss function
        - pred_loss: the energy prediction loss function
        - distant_knn_loss: the k nearest neighbour distance loss function
    '''
    
    model.to(config.device)
    
    log_dir = f'./model_config/{config.log_dir}'
    writer = SummaryWriter(log_dir=log_dir)
    
    # save the config file
    with open(f'{log_dir}/hparams.yaml','w') as f:
        yaml.dump(config,f)
    
    # Initialize early stop object
    early_stop.best_loss = np.Inf
    early_stop.num_epochs_without_improvement = 0

    # convert the numpy array to tensor
    P_tot = torch.from_numpy(P_tot.astype(np.float32))
    Dij = torch.from_numpy(Dij.astype(np.float32))
    ED_ij = torch.from_numpy(ED_ij.astype(int))
    X_j = torch.from_numpy(X_j)
    
    print('\n ------- Start Training -------')
    for epoch in range(config.n_epochs):
        start_time = time.time()
        training_loss = 0
        
        for batch_idx, (x, y, idx) in enumerate(train_loader):  # mini batch
            
            # Configure input
            x = x.to(config.device)
            y = y.to(config.device)
            
            # forward X_j
            model.eval()
            with torch.no_grad():
                batchXj_id = X_j[idx]
                neighbor_input = data_loader.dataset.tensors[0][batchXj_id].reshape(-1, config.input_dim).to(config.device)
                _, _, neighbor_embed, _, _ = model(neighbor_input)
                neighbor_embed = neighbor_embed.reshape(-1, config.knn, neighbor_embed.shape[-1])
                
            # ------------------------------------------
            #  Train VIDA
            # ------------------------------------------
            model.train()
            optimizer.zero_grad()
            
            # get the reconstructed nodes, predicted energy, and the embeddings
            x_recon, y_pred, z, mu, logvar = model(x)
        
            ## compute the total loss
            # vae loss
            recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
            
            # energy prediction loss
            p_loss = pred_loss(y_pred, y)

            # distance loss
            dist_loss = distance_knn_loss(config, z, neighbor_embed, Dij, P_tot, idx, batchXj_id)

            # edit distance loss
            edit_loss = edit_distance_loss(config, z, neighbor_embed, ED_ij, idx)
            
            # scaling the loss
            recon_loss = config.alpha * recon_loss
            kl_loss = config.beta * kl_loss
            p_loss = config.gamma * p_loss
            dist_loss = config.delta * dist_loss
            edit_loss = config.epsilon * edit_loss
            
            # total loss
            loss = recon_loss + kl_loss + p_loss + dist_loss + edit_loss
            
            # backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            
            # ------------------------------------------
            # Log Progress
            # ------------------------------------------
            if batch_idx % config.log_interval == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, config.n_epochs, batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))
                
                writer.add_scalar('training loss',
                                  loss.item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('recon loss',
                                  recon_loss.item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('kl loss',
                                  kl_loss.item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('pred loss',
                                  p_loss.item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('dist loss',
                                  dist_loss.item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('edit loss',
                                  edit_loss.item(),
                                  epoch * len(train_loader) + batch_idx)
    
        print ('====> Epoch: {} Average loss: {:.4f}'.format(epoch, training_loss/len(train_loader.dataset)))
        writer.add_scalar('epoch training loss', training_loss/len(train_loader.dataset), epoch)
        
        # validation
        val_loss, val_bce, val_kld, val_pred, val_dist, val_edit = validate(config, model, data_loader, val_loader, P_tot, Dij, ED_ij, X_j, 
                                                                  vae_loss, pred_loss, distance_knn_loss, edit_distance_loss)
        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('val_recon loss', val_bce, epoch)
        writer.add_scalar('val_kl loss', val_kld, epoch)
        writer.add_scalar('val_pred loss', val_pred, epoch)
        writer.add_scalar('val_dist loss', val_dist, epoch)
        writer.add_scalar('val_edit loss', val_edit, epoch)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        print (f'Epoch {epoch} train+val time: {epoch_time:.2f} seconds \n')
        
        # Check if validation loss has not improved for `patience` epochs
        if early_stop(val_loss, epoch, config.patience):
            break
    
        # Clear the cache
        # torch.mps.empty_cache()
        torch.cuda.empty_cache()
        
        
    writer.close()
    print('\n ------- Finished Training -------')
    
    # save the model
    torch.save(model.state_dict(), f'{log_dir}/model.pt')
    
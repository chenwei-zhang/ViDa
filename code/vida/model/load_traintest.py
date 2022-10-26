# credit to https://github.com/KrishnaswamyLab/GSAE/tree/12d794c27b9c7825ebc1916294c1a390be98fd18

import numpy as np

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import vida.model.eval_metrics

from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score, precision_score, recall_score
from sklearn.neighbors import kneighbors_graph


# load training and test data
def load_trte(train_data,test_data,
              batch_size=32,gnn=False,subsize=None,lognorm=False):

    train_adjs = train_data[0]
    train_coeffs = train_data[1]
    train_energies = train_data[2]
    
    test_adjs = test_data[0]
    test_coeffs = test_data[1]
    test_energies = test_data[2]

    if lognorm:
        # shift
        train_coeffs +=  np.abs(train_coeffs.min()) + 1
        test_coeffs += np.abs(train_coeffs.min()) + 1
        
        # log
        train_coeffs = np.log(train_coeffs)
        test_coeffs = np.log(test_coeffs)

    if gnn:
        train_diracs = torch.eye(train_adjs.shape[-1]).unsqueeze(0).repeat(train_adjs.shape[0],1,1)
        train_tup = (torch.Tensor(train_diracs),
                    torch.Tensor(train_adjs),
                    torch.Tensor(train_energies))
    else:
        train_tup = (torch.Tensor(train_coeffs),
                    torch.Tensor(train_energies))

    if gnn:
        test_diracs = torch.eye(test_adjs.shape[-1]).unsqueeze(0).repeat(test_adjs.shape[0],1,1)
        test_tup = (torch.Tensor(test_diracs),
                    torch.Tensor(test_adjs),
                    torch.Tensor(test_energies))

    else:
        test_tup = (torch.Tensor(test_coeffs), 
                    torch.Tensor(test_adjs), 
                    torch.Tensor(test_energies))
        
    #################
    # SUBSET DATA 
    #################tre
    if subsize != None:
        train_tup, _ = eval_metrics.compute_subsample(train_tup, subsize)
        test_tup, _ = eval_metrics.compute_subsample(test_tup, subsize)

    train_dataset = torch.utils.data.TensorDataset(*train_tup)
    test_dataset = torch.utils.data.TensorDataset(*test_tup)
    
    # get valid set
    train_full_size = len(train_dataset)
    train_split_size = int(train_full_size * .80)
    valid_split_size = train_full_size - train_split_size 
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_split_size, valid_split_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True)

    # valid loader 
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                        shuffle=False)
    
    # early stopping 
    early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=3,
            verbose=True,
            mode='min'
            )
    
    return train_loader, train_tup, test_tup, valid_loader,early_stop_callback

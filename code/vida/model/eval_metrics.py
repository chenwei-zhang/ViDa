# credit to https://github.com/KrishnaswamyLab/GSAE/tree/12d794c27b9c7825ebc1916294c1a390be98fd18

import numpy as np
import networkx as nx

from numpy.random import choice
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score, precision_score, recall_score
from sklearn.neighbors import kneighbors_graph


##############################
# GENERAL UTILS
###############################

def compute_subsample(data, subsize=10000):
    """returns subsampled data 

    Args:
        data ([type]): list of numpy arrays or tensors
        subsize (int, optional): size of sub sample

    Returns:
        (list of subsampled data, sub indies)
    """


    assert type(data) in [list, tuple]
    
    sub_inds = choice(np.arange(data[0].shape[0]), subsize, replace=False)

    sub_data = [x[sub_inds] for x in data]

    return sub_data, sub_inds



def format_metric(array):
    
    array = array.reshape(-1,2)
    
    mean_array = np.round(array.mean(0),3)
    std_array = np.round(array.std(0),3)
    

    return mean_array, std_array 


########################################
# SMOOTHNESS FUNCTIONS
########################################

def get_smoothnes_kNN(embeddings, energies, K):
    """ kNN based graph for smoothness calc

    Args:
        embeddings ([type]): [description]
        energies ([type]): [description]
        K ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    N = embeddings.shape[0]
    
    energies = energies.reshape(N,1)

    # get kNN graph
    print("getting kNN graph")
    A_mat = kneighbors_graph(embeddings, n_neighbors=K, 
                                  mode='connectivity').todense()

    # make symmetric
    A_mat = A_mat + A_mat.T
    A_mat = A_mat.clip(0,1)                         

    
    nn_graph = nx.from_numpy_array(A_mat)
    

    print("computing combinatorial graph laplacian")
    L_mat = nx.laplacian_matrix(nn_graph).todense()
            
    # compute smoothness index
    print("computing smoothness value")
    lap_smooth = np.matmul(L_mat, energies)
    lap_smooth = np.matmul(energies.T, lap_smooth)
    signal_dot = np.matmul(energies.T, energies)
    lap_smooth = lap_smooth/signal_dot
    
    print("smoothness for K={}: {}".format(K, lap_smooth.item()))
    
    return lap_smooth.item()


###################################
# GET SMOOTH VALS 
###################################

def get_smoothness_over_vals(embeddings, energies, smoothness_fn, vals):
    
    smooth_list = []
    for val in vals:
        smooth_indxs = smoothness_fn(embeddings, energies, val)
        smooth_list.append(smooth_indxs)

    return np.array(smooth_list)


def eval_over_replicates(embeddings_reps, energy_reps, 
                        smoothness_fn, val_range):
    """returns an N x M array of smoothness values

    N = number of replicates
    M = number of gamma values used

    Args:
        embeddings_reps ([type]): [description]
        energy_reps ([type]): [description]
        smoothness_fn ([type]): [description]
        val_range ([type]): [description]

    Returns:
        [type]: [description]
    """

    # get dimensions
    n_reps = embeddings_reps.shape[0]
    n_gamma_vals =  embeddings_reps.shape[1]
    n_batch = embeddings_reps.shape[2]
    zdim = embeddings_reps.shape[3]

    eval_rep_embeds = embeddings_reps.reshape(-1,n_batch,zdim)
    print("reshaped eval embedding shape: {}".format(eval_rep_embeds.shape))

    smooth_list = []
    for indx, embed_rep in enumerate(eval_rep_embeds):
        print(indx)

        i = indx // n_gamma_vals

        print("getting smoothness val for replicate: {}".format(i))
        smooth_indxs = []

        for val_i in val_range:
            smooth_val = smoothness_fn(embed_rep,
                                       energy_reps[i],
                                       val_i)
            smooth_indxs.append(smooth_val)

        smooth_list.append(smooth_indxs)
    
    smooth_val_array = np.array(smooth_list)
    

    return np.reshape(smooth_val_array, (n_reps, n_gamma_vals, len(val_range)))

    


#######################
# OTHER METRICS
#######################


def get_metrics(y_pred,y_true):
    
    print(y_pred.shape)
    print(y_true.shape)
    
    roc_auc_met = roc_auc_score(y_true,y_pred)
    
    acc_score = accuracy_score( y_true,y_pred.round())
    
    avgprec = average_precision_score(y_true, y_pred)
    
    prec = precision_score(y_true, y_pred.round())
    recall = recall_score(y_true, y_pred.round())
    
    
    return [roc_auc_met,acc_score,avgprec,prec,recall] 




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

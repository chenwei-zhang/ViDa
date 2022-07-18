import argparse
import numpy as np
from numpy.testing import assert_almost_equal
import copy
import h5py

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from gsae.models.gsae_model import GSAE
from gsae.data_processing.utils import dot2adj
from gsae.data_processing.create_splits import split_data
from gsae.scattering.scattering import transform_dataset, get_normalized_moments
from gsae.utils import eval_metrics


# load dataset generated by Multistrand
def loadtrj(f,FINAL_STRUCTURE,type):
    """load text data and split it into individual trajectory 
    with seperated structure, time, and energy

    Args:
        f: text file with trajectory dp notation, time, energy
            eg. '..((((....)))).', 't=0.000000103', 'seconds, dG=  0.26 kcal/mol\n'
        FINAL_STRUCTURE: final state structure, eg. "..((((....))))."
        type: 'Single' or 'Multiple' mode
    Returns:
        [list]: dot-parenthesis notation, time floats, energy floats
            eg. ['...............', 0.0, 0.0]
    """
    TRAJ=[];i=0;SIM=[]
    
    for s in f:
        ss = s.split(" ",3)
        s_dotparan=ss[0] # dp notation
        s_time = float(ss[1].split("=",1)[1]) # simulation time
        s_energy = float(ss[3].split("=")[1].split("kcal")[0]) # energy
        TRAJ.append([s_dotparan,s_time,s_energy])

        if type == "Single":
            if s_dotparan == FINAL_STRUCTURE: # split to individual trajectory
                SIM.append(TRAJ)
                TRAJ = []
                
    if type == "Multiple":
        SIM = TRAJ
    return SIM


# load multiple trajectories from multiple files
def load_multitrj(folder_name,FINAL_STRUCTURE,num_files):
    # load text file
    SIMS = []; SIMS_concat = []
    for i in range(num_files):
        STR_name = "{}_{}.txt".format(folder_name,i) # PT0
        f = open(STR_name, 'r') # PT0
        SIM = loadtrj(f,FINAL_STRUCTURE,type="Multiple")
        SIMconcat = concat_helix_structures(SIM) 
        SIMS += SIM
        SIMS_concat += SIMconcat
        
    return SIMS,np.array(SIMS),SIMS_concat


# convert concantenate two individual structures to one structure
def concat_helix_structures(SIM):
    """concatenate two individual structures to one structure
    Args:
        SIM: list of individual structures
    Returns:
        SIM_concat: concatenated structure
    """
    SIM_concat = copy.deepcopy(SIM)
    for i in range(len(SIM)):
        if i == 0:
            SIM_concat[i][0] *=2 
        else:
            SIM_concat[i][0] = SIM[i][0].replace("+","")
    return SIM_concat


# assign each states with their labels
def label_structures(SIM_concat,indices):
    """label the visited states of the trajectory 
            based on their unique structure indices
    Args:
        SIM_concat: fully concatenated states of the trajectory
    Returns:
        SIM_dict: fully labeled cstates of the trajectory
    """
    # add a nan column to full states array
    new_col = np.empty(len(np.array(SIM_concat)))
    new_col.fill(np.nan)
    SIM_dict = np.c_[np.array(SIM_concat),new_col]
    # get unique structures
    SIM_dict_uniq = SIM_dict[indices]
    # label the states with its corresponding unique structure indices
    for i in range(len(SIM_dict_uniq)):
        temp = SIM_dict[:,0] == SIM_dict_uniq[i,0]
        indx = np.argwhere(temp==True)
        SIM_dict[indx,3] = i
    return SIM_dict


# convert dot-parenthesis notation to adjacency matrix in a single trajectory
def sim_adj(sim):
    """convert dot-parenthesis notation to adjacency matrix in a single trajectory
    Args:
        sim: [list of sims] dot-parenthesis notation, time floats, energy floats
            eg. [['...............', 0.0, 0.0], ...]
    Returns:
        (tuple): NxN adjacency np matrix, Nx1 energy np array, Nx1 time np array, Nx1 HT np array
    """
    adj_mtr = []
    sim_G = np.array([])
    sim_T = np.array([])
    
    for s in sim:
        sim_T = np.append(sim_T,s[1]) # get time array
        sim_G = np.append(sim_G,s[2]) # get energy array
        
        adj = dot2adj(s[0])
        adj_mtr.append(adj)
    adj_mtr = np.array(adj_mtr) # get adjacency matrix
    
    sim_HT = np.concatenate([np.diff(sim_T),[0]])

    return adj_mtr,sim_G,sim_T,sim_HT


# convert all simulations
def get_whole_data(SIM):
    SIMS_adj=[]; SIMS_G=[]; SIMS_T=[]; SIMS_HT=[]
    for i in range(len(SIM)):
        data = SIM[i]
        SIM_adj,SIM_G,SIM_T,SIM_HT = sim_adj(data)
        assert min(SIM_G) == SIM_G[-1], "Final state is not the minimum energy state."
        SIMS_adj.append(SIM_adj); SIMS_T.append(SIM_T); SIMS_G.append(SIM_G); SIMS_HT.append(SIM_HT)

    SIMS_adj = np.concatenate((SIMS_adj),axis=0)
    SIMS_G = np.concatenate((SIMS_G),axis=0)
    SIMS_T = np.concatenate((SIMS_T),axis=0)
    SIMS_HT = np.concatenate((SIMS_HT),axis=0)
    
    return SIMS_adj,SIMS_G,SIMS_T,SIMS_HT


# get unique structures
def get_unique(SIM_concat,SIM_adj,SIM_G,SIM_T,SIM_HT):
    """
    # get unique states adjacency matrix with their occupancy density
    # get unique energy, and time
    """
    indices, occ_density = uniq_adj_occp(np.array(SIM_concat)[:,0])

    SIM_adj_uniq = SIM_adj[indices]
    SIM_G_uniq = SIM_G[indices]
    SIM_T_uniq = SIM_T[indices]
    SIM_HT_uniq = SIM_HT[indices]
    
    return indices,occ_density,SIM_adj_uniq,SIM_G_uniq,SIM_T_uniq,SIM_HT_uniq


# calulate the occupancy density of each state
def uniq_adj_occp(states):
    """load adjacency matrix and calculate the occupancy density of each state
    Args:
        states: adjacency matrix
    Returns:
        indices,density: indices of unique states, occupancy density of each state
    """
    _, indices, counts = np.unique(states,axis=0,return_index=True,return_counts=True)
    counts = counts[np.argsort(indices)]
    indices = np.sort(indices)
    
    return indices, counts/counts.sum()


# calulate the time fraction of each state
def time_frac(SIM_adj,SIM_adj_uniq,SIM_HT):
    """load time array and calculate the time fraction of each state
    Args:
        SIM_adj,SIM_adj_uniq,SIM_HT
    Returns:
        time fractions: time fraction of each unique state
    """
    time_count = np.zeros(len(SIM_adj_uniq))
    
    for i in range(len(SIM_adj_uniq)):
        for j in range(len(SIM_adj)):
            if np.array_equal(SIM_adj_uniq[i],SIM_adj[j]):
                time_count[i] += SIM_HT[j]  
                
    time_fract = time_count/time_count.sum()
    
    # assert time_count.sum() == SIM_HT.sum(), "Time counts are not equal to total time."
    # assert time_fract.sum() == 1, "Total time fraction is not equal to 1."

    assert_almost_equal(time_count.sum(),SIM_HT.sum(),err_msg='Time counts are not equal to total time.')
    assert_almost_equal(time_fract.sum(),1,err_msg='Total time fraction is not equal to 1.')

    return time_count, time_fract


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


# """save data to h5 file
# """
def save_h5(filename,
            SIM_adj, SIM_scar, SIM_G, SIM_HT,
            SIM_adj_uniq, SIM_scar_uniq, SIM_G_uniq, SIM_HT_uniq,
            occ_density, data_embed, coord_id,
            pca_coords, pca_all_coords,
            phate_coords, phate_all_coords):
    
    hf = h5py.File(filename, "w")
    hf.create_dataset("SIM_adj", data=SIM_adj)
    hf.create_dataset("SIM_scar", data=SIM_scar)
    hf.create_dataset("SIM_G", data=SIM_G)
    hf.create_dataset("SIM_HT", data=SIM_HT)
    hf.create_dataset("SIM_adj_uniq", data=SIM_adj_uniq)
    hf.create_dataset("SIM_scar_uniq", data=SIM_scar_uniq)
    hf.create_dataset("SIM_G_uniq", data=SIM_G_uniq)
    hf.create_dataset("SIM_HT_uniq", data=SIM_HT_uniq)
    # hf.create_dataset("SIM_dict", data=SIM_dict)
    hf.create_dataset("occp", data=occ_density)
    hf.create_dataset("data_embed", data=data_embed)
    hf.create_dataset("coord_id", data=coord_id)
    hf.create_dataset("pca_coords", data=pca_coords)
    hf.create_dataset("pca_all_coords", data=pca_all_coords)
    hf.create_dataset("phate_coords", data=phate_coords)
    hf.create_dataset("phate_all_coords", data=phate_all_coords)    
    hf.close

# if __name__ == "__main__":
#     main()
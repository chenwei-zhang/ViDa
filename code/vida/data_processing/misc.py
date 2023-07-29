import numpy as np
import h5py


# get unique data except for holding time
def get_unique(SIM_concat,SIM_adj,SIM_G,SIM_pair):
    """
    # get unique states adjacency matrix with their occupancy density
    # get unique energy, and time
    """
    indices, occ_density = uniq_adj_occp(np.array(SIM_concat)[:,0])

    SIM_adj_uniq = SIM_adj[indices]
    SIM_G_uniq = SIM_G[indices]
    SIM_pair_uniq = SIM_pair[indices]
    
    return indices,occ_density,SIM_adj_uniq,SIM_G_uniq,SIM_pair_uniq


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


# calulate the average time fraction of unique states
def mean_holdingtime(SIMS_HT, indices_S, coord_id_S):
    """calculate the average time fraction of each unique state
        based on the coordination number: coord_id_S
    """
    SIMS_HT_uniq = np.empty(len(indices_S))
    
    for i in range(len(indices_S)):
        ht_temp = np.where(i==coord_id_S)[0]
        SIMS_HT_uniq[i] = sum(SIMS_HT[ht_temp])/len(ht_temp)

    return SIMS_HT_uniq

# calulate the cumulative time fraction of unique states
def cumu_holdingtime(SIMS_HT, indices_S, coord_id_S):
    """calculate the average time fraction of each unique state
        based on the coordination number: coord_id_S
    """
    SIMS_cumu_uniq = np.empty(len(indices_S))
    cumu_account = np.zeros(len(indices_S),dtype=np.int64)
    
    for i in range(len(indices_S)):
        ht_temp = np.where(i==coord_id_S)[0]
        SIMS_cumu_uniq[i] = sum(SIMS_HT[ht_temp])
        cumu_account[i] = len(ht_temp)

    return SIMS_cumu_uniq, cumu_account


# """save data to h5 file
# """
def save_h5(filename,
            SIMS_G_uniq, SIMS_pair_uniq, occ_density_S, 
            pca_coords, pca_all_coords,
            phate_coords, phate_all_coords,
            umap_coord_2d, umap_all_coord_2d,
            umap_coord_3d, umap_all_coord_3d,
            tsne_coord_2d, tsne_all_coord_2d,
            tsne_coord_3d, tsne_all_coord_3d):
    
    hf = h5py.File(filename, "w")
    hf.create_dataset("SIMS_G_uniq", data=SIMS_G_uniq)
    hf.create_dataset("SIMS_pair_uniq", data=SIMS_pair_uniq)
    hf.create_dataset("occ_density_S", data=occ_density_S)
    
    hf.create_dataset("pca_coords", data=pca_coords)
    hf.create_dataset("pca_all_coords", data=pca_all_coords)
    
    hf.create_dataset("phate_coords", data=phate_coords)
    hf.create_dataset("phate_all_coords", data=phate_all_coords)
    
    hf.create_dataset("umap_coord_2d", data=umap_coord_2d)
    hf.create_dataset("umap_all_coord_2d", data=umap_all_coord_2d)
    
    hf.create_dataset("umap_coord_3d", data=umap_coord_3d)
    hf.create_dataset("umap_all_coord_3d", data=umap_all_coord_3d)
    
    hf.create_dataset("tsne_coord_2d", data=tsne_coord_2d)
    hf.create_dataset("tsne_all_coord_2d", data=tsne_all_coord_2d)
    
    hf.create_dataset("tsne_coord_3d", data=tsne_coord_3d)
    hf.create_dataset("tsne_all_coord_3d", data=tsne_all_coord_3d)  
    
    hf.close
import numpy as np


# freq-weighted distortion
def freq_weighted_distortion(embedding, trj_id):

    def split_embedding(trj_id, embedding):
        traj_embed = []
        split_indx = trj_id+1
        
        for i in range(len(split_indx)):
            if i == 0:
                s = 0
                s_prime = split_indx[i]
            elif i == len(trj_id):
                s = split_indx[i-1]
                s_prime = len(embedding)
            else:
                s = split_indx[i-1]
                s_prime = split_indx[i]
            
            traj_embed.append(embedding[s:s_prime][:,:2])
        
        return traj_embed
    
    def calculate_1trj_distance(arr):
        total_distance = []
        
        for i in range(1, len(arr)):
            total_distance.append(np.sqrt(np.sum((arr[i] - arr[i-1])**2)))
            
        return np.sum(total_distance)
    
    # normalize embedding to [0,1]
    norm_embedding = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding))
    
    # get split embeddings
    traj_embed = split_embedding(trj_id, norm_embedding)
    
    # calculate distortion
    total_distance = 0
    
    for i in range(len(traj_embed)):
        total_distance += calculate_1trj_distance(traj_embed[i])        
    
    distortion = total_distance / (trj_id[-1]+1)
    
    return distortion
       
        



if __name__ == '__main__':
    
    loaded_data = np.load('../../temp/preprocess_Gao-P4T4.npz') 
    indices_all = loaded_data["indices_all"]
    indices_uniq = loaded_data["indices_uniq"]

    loaded_data = np.load('../../temp/time_Gao-P4T4.npz')
    trj_id = loaded_data["trj_id"]


    loaded_data = np.load('../../temp/embed_Gao-P4T4.npz')
    pca_coords_uniq = loaded_data["pca_coords_uniq"]
    phate_coords_uniq = loaded_data["phate_coords_uniq"]

    pca_coords = pca_coords_uniq[indices_all]
    phate_coords = phate_coords_uniq[indices_all]


    print(f"ViDa PCA: {freq_weighted_distortion(pca_coords, trj_id):.3f}")
    print(f"ViDa PHATE: {freq_weighted_distortion(phate_coords, trj_id):.3f}")




# ## sanity check ##
# # load ViDa embedding plot data
# fnpz_data_embed = f"../../code/data/vida_data/PT4_0823-0138.npz"
# data_npz_embed = np.load(fnpz_data_embed,allow_pickle=True)
# # asssign data to variables
# for var in data_npz_embed.files:
#     globals()[var] = data_npz_embed[var]
#     print(var, globals()[var].shape)
# print(f"ViDa PCA: {freq_weighted_distortion(pca_all_coords, trj_id):.3f}")
# print(f"ViDa PHATE: {freq_weighted_distortion(phate_all_coords, trj_id):.3f}")

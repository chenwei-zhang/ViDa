import numpy as np
import pickle
import argparse


def sim_ht(sim_T):
    """calculate holding time for each trajectory
    """
    sim_HT = np.array([])
    idx = np.where(sim_T==0)[0]
    
    for i in range(len(idx)):
        if i < len(idx)-1:
            temp_T = sim_T[idx[i]:idx[i+1]]
            sim_HT = np.append(sim_HT,np.concatenate([np.diff(temp_T),[0]]))
        else:
            temp_T = sim_T[idx[i]:]
            sim_HT = np.append(sim_HT,np.concatenate([np.diff(temp_T),[0]]))
    
    # get each individual trajectory's index
    trj_id = np.where(sim_HT==0)[0]

    return sim_HT, trj_id


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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', required=True, help='preprocessed data file')
    parser.add_argument('--outpath', required=True, help='output time data')

    args = parser.parse_args()

    inpath = args.inpath
    outpath = args.outpath
    
    # Load the data
    print(f"[comp_time] Loading preprocessed SIMS_T and index from {inpath}")

    with open(inpath, 'rb') as file:
        loaded_data = pickle.load(file)
    
    SIMS_T = loaded_data["SIMS_T"]
    indices_S = loaded_data["indices_S"]
    coord_id_S = loaded_data["coord_id_S"]
    
    # calculate holding time for each trajectory
    print("[comp_time] Calculating holding time for each trajectory")

    SIMS_HT, trj_id = sim_ht(SIMS_T)
    
    # calculate the average (unique) holding time
    print("[comp_time] Calculating the average holding time for each unique state")
    
    SIMS_HT_uniq = mean_holdingtime(SIMS_HT, indices_S, coord_id_S)


    # calculate the cumulative (unique) holding time
    print("[comp_time] Calculating the cumulative holding time for each unique state")

    SIMS_cumu_HT_uniq,cumu_account_uniq = cumu_holdingtime(SIMS_HT, indices_S, coord_id_S)

    # save time data
    print(f"[comp_time] Saving time data to {outpath}")
    
    data_to_save = {
    "SIMS_HT": SIMS_HT,
    "SIMS_HT_uniq": SIMS_HT_uniq,
    "SIMS_cumu_HT_uniq": SIMS_cumu_HT_uniq,
    "cumu_account_uniq": cumu_account_uniq,
    "trj_id": trj_id,
    }
    
    with open(outpath, 'wb') as file:
        pickle.dump(data_to_save, file)
        

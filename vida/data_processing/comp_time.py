import numpy as np
import argparse
import time


def sim_ht(trans_time, Machineck=False):
    """calculate holding time for each trajectory
    """
    hold_time = np.array([])
    idx = np.where(trans_time==0)[0]
    
    for i in range(len(idx)):
        if i < len(idx)-1:
            temp_t = trans_time[idx[i]:idx[i+1]]
            hold_time = np.append(hold_time,np.concatenate([np.diff(temp_t),[0]]))
        else:
            temp_t = trans_time[idx[i]:]
            hold_time = np.append(hold_time,np.concatenate([np.diff(temp_t),[0]]))
    
    if Machineck:
        temp = np.append(idx, len(trans_time))
        trj_id = (temp-1)[1:]
    else: 
        # get each individual trajectory's index
        trj_id = np.where(hold_time==0)[0]

    return hold_time, trj_id


# calulate the average time fraction of unique states
def mean_holdingtime(hold_time, indices_uniq, indices_all):
    """calculate the average time fraction of each unique state
        based on the coordination number: indices_all
    """
    hold_time_uniq = np.empty(len(indices_uniq))
    
    for i in range(len(indices_uniq)):
        ht_temp = np.where(i==indices_all)[0]
        hold_time_uniq[i] = sum(hold_time[ht_temp])/len(ht_temp)

    return hold_time_uniq


# calulate the cumulative time fraction of unique states
def cumu_holdingtime(hold_time, indices_uniq, indices_all):
    """calculate the average time fraction of each unique state
        based on the coordination number: indices_all
    """
    cum_time_uniq = np.empty(len(indices_uniq))
    freq_uniq = np.zeros(len(indices_uniq),dtype=np.int64)
    
    for i in range(len(indices_uniq)):
        ht_temp = np.where(i==indices_all)[0]
        cum_time_uniq[i] = sum(hold_time[ht_temp])
        freq_uniq[i] = len(ht_temp)

    return cum_time_uniq, freq_uniq




if __name__ == '__main__':
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', required=True, help='preprocessed data file')
    parser.add_argument('--outpath', required=True, help='output time data')

    args = parser.parse_args()

    inpath = args.inpath
    outpath = args.outpath
    
    # Load the data
    print(f"[Comp_time] Loading preprocessed trans_time and index from {inpath}")

    loaded_data = np.load(inpath)
    
    trans_time = loaded_data["trans_time"]
    indices_uniq = loaded_data["indices_uniq"]
    indices_all = loaded_data["indices_all"]
    
    # calculate holding time for each trajectory
    print("[Comp_time] Calculating holding time for each trajectory")

    # TODO
    hold_time, trj_id = sim_ht(trans_time, Machineck=True)
    
    # calculate the average (unique) holding time
    print("[Comp_time] Calculating the average holding time for each unique state")
    
    hold_time_uniq = mean_holdingtime(hold_time, indices_uniq, indices_all)


    # calculate the cumulative (unique) holding time
    print("[Comp_time] Calculating the cumulative holding time for each unique state")

    cum_time_uniq,freq_uniq = cumu_holdingtime(hold_time, indices_uniq, indices_all)

    # save time data
    print(f"[Comp_time] Saving time data to {outpath}")
    
    data_to_save = {
    "hold_time": hold_time,
    "hold_time_uniq": hold_time_uniq,
    "cum_time_uniq": cum_time_uniq,
    "freq_uniq": freq_uniq,
    "trj_id": trj_id,
    }
    
    np.savez_compressed(outpath, **data_to_save)

    print("[Comp_time] Done!")
    
    # Record the end time
    end_time = time.time()
    
    # Print the time elapsed
    print(f"[Comp_time] Elapsed Time: {(end_time - start_time):.3f} seconds")
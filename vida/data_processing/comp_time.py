import numpy as np
import argparse
import time
import tqdm


def empirical_holding_times(arrival_times):
    """
    calculate the empirical holding time of each state, along each trajectory
    """

    # get each individual trajectory's index
    starts = np.where(arrival_times==0)[0]
    ends = np.append(starts[1:]-1, len(arrival_times)-1)
    
    hold_time = np.array([])
    for i in tqdm.tqdm(range(len(starts))):
        if i < len(starts)-1:
            times = arrival_times[starts[i]:starts[i+1]]
        else:
            times = arrival_times[starts[i]:]

        hold_time = np.append(hold_time,np.concatenate([np.diff(times),[0]])) 

    return hold_time, ends


# calulate the average time fraction of unique states
def mean_holdingtime(hold_time, indices_uniq, indices_all):
    """calculate the average time fraction of each unique state
        based on the coordination number: indices_all
    """

    cum_time_uniq, freq_uniq = cumu_holdingtime(hold_time, indices_uniq, indices_all)
    return cum_time_uniq/freq_uniq


# calulate the cumulative time fraction of unique states
def cumu_holdingtime(hold_time, indices_uniq, indices_all):
    """ 
    empirical holding time of each unique state
    """
    cum_time_uniq = np.zeros(len(indices_uniq),dtype=float)
    freq_uniq = np.zeros(len(indices_uniq),dtype=int)

    np.add.at(cum_time_uniq, indices_all, hold_time)
    np.add.at(freq_uniq, indices_all, 1)

    return cum_time_uniq, freq_uniq




if __name__ == '__main__':
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--rxn', required=True, help='Reaction name')

    args = parser.parse_args()

    reaction_id = args.rxn

    inpath = "data/post_data/{}/preprocess_{}.npz".format(reaction_id, reaction_id)
    outpath = "data/post_data/{}/time_{}.npz".format(reaction_id, reaction_id)


    # Load the data
    print(f"[Comp_time] Loading preprocessed trans_time and index from {inpath}")

    loaded_data = np.load(inpath)
    
    trans_time = loaded_data["trans_time"]
    indices_uniq = loaded_data["indices_uniq"]
    indices_all = loaded_data["indices_all"]
    
    # calculate holding time for each trajectory
    print("[Comp_time] Calculating holding time for each trajectory")

    # get the holding time for each trajectory
    hold_time, endpoints = empirical_holding_times(trans_time)

    # calculate the cumulative (unique) holding time
    print("[Comp_time] Calculating the cumulative holding time for each unique state")

    cum_time_uniq, freq_uniq = cumu_holdingtime(hold_time, indices_uniq, indices_all)

    # calculate the average (unique) holding time
    print("[Comp_time] Calculating the average holding time for each unique state")
    hold_time_uniq = cum_time_uniq/freq_uniq

    # save time data
    print(f"[Comp_time] Saving time data to {outpath}")
    
    data_to_save = {
    "hold_time": hold_time,
    "hold_time_uniq": hold_time_uniq,
    "cum_time_uniq": cum_time_uniq,
    "freq_uniq": freq_uniq,
    "trj_id": endpoints,
    }
    
    np.savez_compressed(outpath, **data_to_save)

    print("[Comp_time] Done!")
    
    # Record the end time
    end_time = time.time()
    
    # Print the time elapsed
    print(f"[Comp_time] Elapsed Time: {(end_time - start_time):.3f} seconds")
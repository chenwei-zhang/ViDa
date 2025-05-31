import numpy as np
import copy
import re
import tqdm



# get the unique structures and their corresponding indices
def get_uniq(dp, dp_og, energy, order_cid=None, pair=None):
        
    dp_og_uniq, indices_uniq, indices_all = np.unique(dp_og,return_index=True,return_inverse=True)
    
    dp_uniq = dp[indices_uniq]
    energy_uniq = energy[indices_uniq]
    
    if pair is not None:
        pair_uniq = pair[indices_uniq]
    else:
        pair_uniq = None
        
    if order_cid is not None:
        cid_uniq = order_cid[indices_uniq]
        id_uniq = []
        for cid in tqdm.tqdm(cid_uniq, total=len(cid_uniq)):
            id_uniq.append(parse_cids(cid.encode(encoding='utf-8').split(b' ')))
        id_uniq = np.array(id_uniq, dtype=object)
    else:
        id_uniq = None

    return dp_uniq, dp_og_uniq, energy_uniq, id_uniq, pair_uniq, indices_uniq, indices_all

       

def read_machinek(fpath, num_traj):
    def _read_trajectory_h5(fpath, sim_no): 
        import h5py as h5
        with h5.File(fpath, "r") as f:
            times = f[str(sim_no)]["times"][:]
            energies = f[str(sim_no)]["energies"][:]
            structs = [s.decode() for s in f[str(sim_no)]["structs"]]          
            ids = [s.decode() for s in f[str(sim_no)]["ordered_ids"]]
        return times, energies, structs, ids
        
    trajs_states, trajs_times, trajs_energies, trajs_ids  = [],[],[], []

    for i in tqdm.tqdm(range(num_traj)):
        traj = _read_trajectory_h5(fpath, i)
        trajs_times.append(traj[0])
        trajs_energies.append(traj[1])
        trajs_states.append(traj[2])
        trajs_ids.append(traj[3])
    
    trajs_times = np.array(trajs_times, dtype=object)
    trajs_energies = np.array(trajs_energies, dtype=object)
    trajs_states = np.array(trajs_states, dtype=object)
    trajs_ids = np.array(trajs_ids, dtype=object)
        
    return trajs_states, trajs_times, trajs_energies, trajs_ids



# cooncatanate all sturcutres for machinek dataset: 
def concat_machinek(states, times, energies, trajs_ids):
    # convert concantenate two individual structures to one structure 
    def process_machinek(dp_og):
        dp = copy.deepcopy(dp_og)
        for i in range(len(dp)):
            if " " in dp[i]:
                dp[i] = dp[i].replace(" ","").replace("+","")                
            else:
                dp[i] = dp[i].replace("+","")
                
        return np.array(dp)

    dp, dp_og, energy, trans_time, order_cid = [],[],[],[],[]
    
    for i in tqdm.tqdm(range(len(states))):
        sims_dp = process_machinek(states[i])
        dp.append(sims_dp)
        dp_og.append(states[i])
        energy.append(energies[i])
        trans_time.append(times[i])
        order_cid.append(trajs_ids[i])
        
    dp = np.concatenate(dp)
    dp_og = np.concatenate(dp_og)
    energy = np.concatenate(energy)
    trans_time = np.concatenate(trans_time)
    order_cid = np.concatenate(order_cid)
        
    return dp, dp_og, energy, trans_time, order_cid

 
# assign unique identifier to each base
def assign_base_names(sequence):
    split_sequence = re.split(r'\s|\+', sequence)
    base_names = []

    for strand_index, strand in enumerate(split_sequence):
        strand_names = []
        
        for base_index, base_type in enumerate(strand):
            strand_names.append(f'{chr(ord("a") + strand_index)}{base_index + 1}')
            
        base_names.append(strand_names)
    
    return base_names


def parse_cids(cids):
    # decode strand IDs
    sids = [np.array(c.split(b'+'), dtype=int) // 2 for c in cids]
    smin = np.concatenate(sids).min()
    sids = [(c - smin).tolist() for c in sids]
    # find minimal cyclic permutation per complex
    prm_parts = lambda c: lambda p: (c[p:len(c)], c[0:p])
    prm = lambda c: lambda p: sum(prm_parts(c)(p), start=[])
    argprm = lambda c: min(np.atleast_1d(np.argmin(c)), key=prm(c))
    return [tuple(prm(c)(argprm(c))) for c in sids]
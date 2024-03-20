import numpy as np
import copy
import re
from strandReorder import ThreeStrandReorder


# load dataset generated by Multistrand
def read_1trj(f):
    """load text data and split it into individual trajectory 
    with seperated structure, time, and energy

    Args:
        f: text file with trajectory dp notation, time, energy, and if paired (1) or not (0)
            eg. '..((((....)))).', 't=0.000000103', 'seconds, dG=  0.26 kcal/mol\n', "0"
        FINAL_STRUCTURE: final state structure, eg. "..((((....))))."
        type: 'Single' or 'Multiple' mode
    Returns:
        [list]: dot-parenthesis notation, time floats, energy floats, paired or not
            eg. ['...............', 0.0, -12.0, 1]
    """
    dp_list = []; time_list = []; energy_list = []; pair_list = []
    
    for s in f:
        ss = s.split(" ")
        s_dp=ss[0] # dp notation
        s_time = float(ss[1].split("=")[1]) # simulation time
        s_energy = float(ss[3].split("=")[1]) # energy
        s_pair = int(ss[-1]) # paired or not
        
        dp_list.append(s_dp)
        time_list.append(s_time)
        energy_list.append(s_energy)
        pair_list.append(s_pair)
        
    return [dp_list,time_list,energy_list,pair_list]


# load multiple trajectories from multiple files
def read_gao(fpath,rxn,num_files=100):
    trajs_states, trajs_times, trajs_energies, trajs_pairs = [],[],[],[]
    
    for i in range(num_files):
        sim_name = f"{fpath}/{rxn}/{rxn}-{i}.txt"
        f = open(sim_name, 'r') 
        trj = read_1trj(f)
        trajs_states.append(trj[0])
        trajs_times.append(trj[1])
        trajs_energies.append(trj[2])
        trajs_pairs.append(trj[3])
    
    trajs_states = np.array(trajs_states, dtype=object)
    trajs_times = np.array(trajs_times, dtype=object)
    trajs_energies = np.array(trajs_energies, dtype=object)
    trajs_pairs = np.array(trajs_pairs, dtype=object)
        
    return trajs_states, trajs_times, trajs_energies, trajs_pairs


# convert concantenate two individual structures to one structure 
def process_gao(dp_og):
    dp = copy.deepcopy(dp_og)
    for i in range(len(dp)):   
        dp[i] = dp[i].replace("+","")
    
    return dp


# convert concantenate two individual structures to one structure, and pair or not
def process_hata(dp_og):
    dp = copy.deepcopy(dp_og)
    dp_pair = []
    for i in range(len(dp)):
        if "&" in dp[i]:
            dp[i] = dp[i].replace("&","")
            dp_pair.append(0)
            
        if "+" in dp[i]:
            dp[i] = dp[i].replace("+","")
            dp_pair.append(1)
            
    return np.array(dp), np.array(dp_pair)


# cooncatanate all sturcutres for Gao dataset: 
def concat_gao(states, times, energies, pairs):
    
    dp, dp_og, pair, energy, trans_time = [],[],[],[],[]
    
    for i in range(len(states)):
        sims_dp = process_gao(states[i])
        
        dp.append(sims_dp)
        dp_og.append(states[i])
        trans_time.append(times[i])
        energy.append(energies[i])
        pair.append(pairs[i])
    
    dp = np.concatenate(dp)
    dp_og = np.concatenate(dp_og)
    pair = np.concatenate(pair)
    energy = np.concatenate(energy)
    trans_time = np.concatenate(trans_time)
        
    return dp, dp_og, pair, energy, trans_time


# cooncatanate all sturcutres for Hata dataset:: 
def concat_hata(states, times, energies):
    
    dp, dp_og, pair, energy, trans_time = [],[],[],[],[]
    
    for i in range(len(states)):
        sims_dp, sims_pair = process_hata(states[i])

        dp.append(sims_dp)
        dp_og.append(states[i])
        pair.append(sims_pair)
        energy.append(energies[i])
        trans_time.append(times[i])
    
    dp = np.concatenate(dp)
    dp_og = np.concatenate(dp_og)
    pair = np.concatenate(pair)
    energy = np.concatenate(energy)
    trans_time = np.concatenate(trans_time)
        
    return dp, dp_og, pair, energy, trans_time


# get the unique structures and their corresponding indices
def get_uniq(dp, dp_og, pair, energy):
    
    if dp[0].ndim > 0:
        dp = np.concatenate(dp)
    
    indices_uniq = np.unique(dp_og,return_index=True)[1]
    
    dp_uniq = dp[indices_uniq]
    dp_og_uniq = dp_og[indices_uniq]
    pair_uniq = pair[indices_uniq]
    energy_uniq = energy[indices_uniq]
        
    # find index to recover to all data from unique data
    indices_all = np.empty(len(dp))
    for i in range(len(dp_uniq)):
        temp = dp == dp_uniq[i]
        indx = np.argwhere(temp==True)
        indices_all[indx] = i
    indices_all = indices_all.astype(int)

    return dp_uniq, dp_og_uniq, pair_uniq, energy_uniq, indices_uniq, indices_all



# label the structural types
def label_struc(trajs_types, dp_og_uniq):
    
    type_uniq = []
    
    for i in range(len(dp_og_uniq)):
        type_uniq.append(trajs_types[dp_og_uniq[i]])
    type_uniq = np.array(type_uniq)
    
    return type_uniq


# three-strand readout
def readout_3strand(lines, ref_name_list, strand_list, strand_sub, strand_incb, strand_inv):
    
    reoder_3strand = ThreeStrandReorder()

    seq_line = []; dp_list = []; time_list = []; energy_list = []; shortname_list = []; incb_inv_pair_list = []
    
    pattern = re.compile(r'^(.*?)\s+t=([\d.e+-]+) seconds, dG=([-\d.e+]+) kcal/mol')

    for i in range(len(lines)): 
        if strand_sub in lines[i]:
            sequence = lines[i]
        
        match = re.search(pattern, lines[i])
        if match:
            dp = match.group(1)
            time = float(match.group(2))
            energy = float(match.group(3))
            
            dp_new, short_seqname, incb_inv_pair = reoder_3strand.dp_reorder(
                                                        dp,
                                                        sequence, 
                                                        ref_name_list,
                                                        strand_list,
                                                        strand_sub, 
                                                        strand_incb, 
                                                        strand_inv,
                                                        )
            seq_line.append(sequence)        
            dp_list.append(dp_new)
            time_list.append(time)
            energy_list.append(energy)
            shortname_list.append(short_seqname)
            incb_inv_pair_list.append(incb_inv_pair)
             
    return (seq_line, dp_list, time_list, energy_list, shortname_list, incb_inv_pair_list)
        
        
def read_machinek(fpath, rxn, ref_name_list, strand_list, strand_sub, strand_incb, strand_inv, num_files):
    trajs_seqs, trajs_states, trajs_times, trajs_energies, trajs_shortnames, trajs_incbinvpair = [],[],[],[],[],[]
    
    for i in range(num_files):
        sim_name = f"{fpath}/{rxn}-{i}.txt"
        f = open(sim_name, 'r')
        lines = f.read().splitlines()
        trj = readout_3strand(lines, ref_name_list, strand_list, strand_sub, strand_incb, strand_inv)
        trajs_seqs.append(trj[0])
        trajs_states.append(trj[1])
        trajs_times.append(trj[2])
        trajs_energies.append(trj[3])
        trajs_shortnames.append(trj[4])
        trajs_incbinvpair.append(trj[5])
    
    trajs_seqs = np.array(trajs_seqs, dtype=object)
    trajs_states = np.array(trajs_states, dtype=object)
    trajs_times = np.array(trajs_times, dtype=object)
    trajs_energies = np.array(trajs_energies, dtype=object)
    trajs_shortnames = np.array(trajs_shortnames, dtype=object)
    trajs_incbinvpair = np.array(trajs_incbinvpair, dtype=object)
    
    return trajs_seqs, trajs_states, trajs_times, trajs_energies, trajs_shortnames, trajs_incbinvpair
        

# cooncatanate all sturcutres for machinek dataset: 
def concat_machinek(states, times, energies):
    # convert concantenate two individual structures to one structure 
    def process_machinek(dp_og):
        dp = copy.deepcopy(dp_og)
        dp_pair = []
        for i in range(len(dp)):
            if " " in dp[i]:
                dp[i] = dp[i].replace(" ","").replace("+","")
                dp_pair.append(0)
                
            else:
                dp[i] = dp[i].replace("+","")
                dp_pair.append(1)
                
        return np.array(dp), np.array(dp_pair)

    dp, dp_og, pair, energy, trans_time = [],[],[],[],[]
    
    
    for i in range(len(states)):
        sims_dp, sims_pair = process_machinek(states[i])

        dp.append(sims_dp)
        dp_og.append(states[i])
        pair.append(sims_pair)
        energy.append(energies[i])
        trans_time.append(times[i])
    
    dp_arr = np.array(dp, dtype=object)
    dp_og = np.concatenate(dp_og)
    pair = np.concatenate(pair)
    energy = np.concatenate(energy)
    trans_time = np.concatenate(trans_time)
        
    return dp_arr, dp_og, pair, energy, trans_time
    
 
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
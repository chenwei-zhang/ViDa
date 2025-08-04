from multistrand.objects import *
from multistrand.options import Options, Literals
from multistrand.system import SimSystem
from multistrand.utils.utility import printTrajectory, normalizeCyclicPermutation

import numpy as np
import pandas as pd
import h5py as h5
import argparse
import sys

from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--row", type=int)
parser.add_argument("--nsims", default=1, type=int)
parser.add_argument("--save_id", default="", type=str)
args = parser.parse_args()


def load_Gao_row(data_filename, row): 
   
    df = pd.read_csv(data_filename)
    row_data = df.loc[row]

    sequences = {"P4": row_data["P4 sequence (5'->3')"],
                 "T4": row_data["T4 sequence (5'->3')"],}
    
    specs =  {"temperature": row_data["temperature (°C)"],
              "sodium": row_data["sodium conc (M)"],
              "magnesium": row_data["magnesium conc (M)"],
              "join_concentration": 1e-9*row_data["P4 conc (nM)"]}

    expid = row_data["Experiment ID"] + "_conc"+str(row_data["T4 conc (nM)"])

    return expid, sequences, specs


def create_options_FSM(sequences, experiment_specs, simulation_specs):
                       
    opt = Options(
        simulation_mode = Literals.first_step,
        **experiment_specs, 
        **simulation_specs)

    opt.DNA23Metropolis()
    
    p4 = Strand(name="P4", sequence=sequences["P4"])
    t4 = Strand(name="T4", sequence=sequences["T4"])
    initial_p4 = Complex(strands=[p4], structure="."*len(p4.sequence), boltzmann_sample=True)
    initial_t4 = Complex(strands=[t4], structure="."*len(t4.sequence), boltzmann_sample=True)
    success_complex = Complex(strands=[p4, t4],structure="("*len(p4.sequence) + "+" + ")"*len(t4.sequence))
    stop_success = StopCondition(Literals.success, [(success_complex, Literals.exact_macrostate, 0)])
    failed_complex = Complex(strands=[p4],structure="."*len(p4.sequence))
    stop_fail = StopCondition(Literals.failure, [(failed_complex, Literals.dissoc_macrostate, 0)])
    
    # onedomain = Domain(name="itall",sequence=sequences["P4"])
    # p4 = Strand(name="P4",domains=[onedomain])
    # t4 = p4.C
    # initial_p4 = Complex(strands=[p4],structure=".", boltzmann_sample=True)
    # initial_t4 = Complex(strands=[t4],structure=".", boltzmann_sample=True)
    # success_complex = Complex(strands=[p4, t4], structure="(+)")
    # stop_success = StopCondition(Literals.success, [(success_complex, Literals.exact_macrostate, 0)])
    # failed_complex = Complex(strands=[p4], structure=".")
    # stop_fail = StopCondition(Literals.failure, [(failed_complex, Literals.dissoc_macrostate, 0)])

    opt.start_state = [initial_p4, initial_t4]
    opt.stop_conditions = [stop_success, stop_fail]  

    return opt



def bootstrap_options(opt, n, t):

    opt_new = opt.restart_from_checkpoint(n-1)

    opt_new.simulation_mode = Literals.trajectory
    opt_new.simulation_time = t
    
    del opt
    
    return opt_new


def save_sim(sim_no, traj_filename, structs, energies, times, traj_seed, ordered_ids):

    with h5.File(traj_filename, "a") as f:

        grp = f.create_group(str(sim_no))

        dtimes = grp.create_dataset("times", data=times, dtype=np.float64, maxshape = (None,))
        denergies = grp.create_dataset("energies", data=energies, dtype=np.float64, maxshape = (None,))
        dstructs = grp.create_dataset("structs", data=[s.encode() for s in structs], dtype=h5.string_dtype(), maxshape = (None,))
        dtraj_seed = grp.create_dataset("interface_trajectory_seed", data=traj_seed)
        dordered_ids = grp.create_dataset("ordered_ids", data=[" ".join(["+".join([str(c) for c in cmplx]) for cmplx in state]).encode() for state in ordered_ids], dtype=h5.string_dtype(), maxshape = (None,))
        

def append_sim(sim_no, traj_filename, structs, energies, times, ordered_ids):

    with h5.File(traj_filename, "a") as f:

        # offset by 1 to omit the repeated initial state

        n = f[str(sim_no)]["times"].shape[0]
        m = times.shape[0]

        f[str(sim_no)]["times"].resize((n + m - 1), axis = 0)
        f[str(sim_no)]["times"][-m+1:] = times[1:]

        f[str(sim_no)]["energies"].resize((n + m - 1), axis = 0)
        f[str(sim_no)]["energies"][-m+1:] = energies[1:]

        f[str(sim_no)]["structs"].resize((n + m - 1), axis = 0)
        f[str(sim_no)]["structs"][-m+1:] = [s.encode() for s in structs[1:]]

        f[str(sim_no)]["ordered_ids"].resize((n + m - 1), axis = 0)
        f[str(sim_no)]["ordered_ids"][-m+1:] = [" ".join(["+".join([str(c) for c in cmplx]) for cmplx in state]).encode() for state in ordered_ids[1:]]

        maxidx = f[str(sim_no)]["structs"].shape[0] - 1

    return maxidx


def save_checkpoint(opt, sim_no, traj_filename, n, idx):

    cp_log = opt.full_trajectory[n-1]

    assert len(cp_log)==1

    with h5.File(traj_filename, "a") as f:

        grp = f[str(sim_no)]
        subgrp = grp.create_group("checkpoint"+str(idx))

        dend_seed =  subgrp.create_dataset("seed", data= cp_log[0].seed)
        dend_strand_names =  subgrp.create_dataset("strand_names", data= cp_log[0].strand_names)
        dend_structure =  subgrp.create_dataset("structure", data= cp_log[0].structure)
        dtime =  subgrp.create_dataset("time", data= opt.full_trajectory_times[n-1])


def read_sim(traj_filename, sim_no): 
    with h5.File(traj_filename, "r") as f:
        times = f[str(sim_no)]["times"][:]
        energies = f[str(sim_no)]["energies"][:]
        structs = [s.decode() for s in f[str(sim_no)]["structs"]]
        trajectory_seed = f[str(sim_no)]["interface_trajectory_seed"][()]
        ordered_ids = f[str(sim_no)]["ordered_ids"][()]
    return structs, energies, times, trajectory_seed


def pack_trajectory(opt: Options):

    strands = opt.strand_names()

    ordered_structs = []
    ordered_ids = []

    for state in opt.full_trajectory:

        ids = [[strands[n].id for n in cmplx.strand_names.split(',')] for cmplx in state]

        order = np.argsort([min(s) for s in ids]) # inducing an ordering of the complexes 

        normalized = [normalizeCyclicPermutation(ids[i], state[i].sequence, state[i].structure) for i in order]
        seq = [s[0] for s in normalized]
        sct = [s[1] for s in normalized]

        ordered_structs.append(sct)
        ordered_ids.append([ids[i] for i in order])

    structs = np.array(
        [' '.join(s) for s in ordered_structs],
        dtype=str)
    energies = np.array(
        [sum(cmplx.energy for cmplx in s) for s in opt.full_trajectory],
        dtype=np.float64)
    times = np.array(opt.full_trajectory_times, dtype=np.float64)
    end = opt.interface.end_states
    return (structs, energies, times, end, ordered_ids)


def incomplete(stoplog): 
    a = sorted([item.split(":")[-1] for item in stoplog[0][0].strand_names.split(",")])

    return a == ['P4', 'T4']


def simulate(opt: Options, verbose: bool=False):
    sys = SimSystem(opt)
    sys.start()
    if verbose:
        printTrajectory(opt, show_seed=True)
    
    return pack_trajectory(opt)


def batched_simulation(opt, sim_no, traj_filename):

    structs, energies, times, stoplog, ordered_ids = simulate(opt, verbose=True)

    save_sim(sim_no, traj_filename, structs, energies, times, opt.interface_trajectory_seed, ordered_ids)

    simulation_time = opt.simulation_time
    idx = len(structs)

    # while incomplete(stoplog): 

    #     save_checkpoint(opt, sim_no, traj_filename, len(structs), idx)
    #     opt = bootstrap_options(opt, len(structs), times[-1]+simulation_time)

    #     structs, energies, times, stoplog, ordered_ids = simulate(opt, verbose=False)
    #     idx = append_sim(sim_no, traj_filename, structs, energies, times, ordered_ids)

    

def main():

    row = args.row
    nsims = args.nsims
    save_id = "" if args.save_id==None else "_"+args.save_id 


    simulation_specs = {"num_simulations":1,
                        "output_interval":1,
                        "simulation_time":float('inf'),
                        }


    data_filename = "reactions/Gao_data.csv"
    expid, sequences, experiment_specs = load_Gao_row(data_filename, row)
    

    traj_filename = "reactions/gao/" + expid +"/" + expid+save_id + ".hdf5"
    assert not Path(traj_filename).exists()
    
    for sim_no in range(nsims):
        print("\nStarting simulation", sim_no)
        sys.stdout.flush()
        
        opt = create_options_FSM(sequences, experiment_specs, simulation_specs)
        batched_simulation(opt, sim_no, traj_filename)
        
        print("Finished simulation", sim_no)
        sys.stdout.flush()

if __name__ == "__main__":
    main()



# Example: from the machinek directory run
    # python sample_trajs_gao.py --row 0 --nsims 10 --save_id test

# Arguments: 
    # "row" is the reaction to pull from the csv file 
    # "nsims" is the the number of FSM samples
    # save_id is appended to the experiment id when the hdf5 file is saved
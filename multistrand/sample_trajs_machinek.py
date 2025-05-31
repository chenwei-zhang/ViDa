from multistrand.objects import *
from multistrand.options import Options, Literals
from multistrand.system import SimSystem
from multistrand.utils.utility import printTrajectory, normalizeCyclicPermutation

import numpy as np
import pandas as pd
import h5py as h5
import argparse

from pathlib import Path
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--row", type=int)
parser.add_argument("--nsims", default=1, type=int)
parser.add_argument("--save_id", default="", type=str)
parser.add_argument("--timestep", default=1e-7, type=float)
args = parser.parse_args()


def load_Machinek_row(data_filename, row): 
   
    df = pd.read_csv(data_filename)
    row_data = df.loc[row]
    row_data = df.loc[row]

    sequences = {"incumbent": row_data["Incumbent sequence (5'->3')"],
                 "target": row_data["Target sequence (5'->3')"],
                 "invader": row_data["Invader sequence (5'->3')"]}
    
    specs =  {"temperature": row_data["temperature (°C)"],
              "sodium": row_data["sodium conc (M)"],
              "magnesium": row_data["magnesium conc (M)"],
              "join_concentration": 1e-9*row_data["Invader conc (nM)"]}

    expid = row_data["Experiment ID"] + "_conc"+str(row_data["Invader conc (nM)"])

    return expid, sequences, specs


def create_options_FSM(sequences, experiment_specs, simulation_specs):
                       
    opt = Options(
        simulation_mode = Literals.first_step,
        **experiment_specs, 
        **simulation_specs)

    opt.DNA29Arrhenius()

    incumbent = Strand(name="incumbent", sequence=sequences["incumbent"])       # Assigned ID 0 by MS
    target    = Strand(name="target",    sequence=sequences["target"])          # Assigned ID 2 by MS
    invader   = Strand(name="invader",   sequence=sequences["invader"])         # Assigned ID 4 by MS

    
    intialDotParen = '.' * 16 + '(' * 17 + "+" + '.' * 10 + ')' * 17  
    # initial_duplx = Complex(strands=[incumbent, target],
    #                         structure=len(incumbent.sequence)*"(" + 
    #                                   "+" + 
    #                                   (len(target.sequence)-len(incumbent.sequence))*"." + 
    #                                   len(incumbent.sequence)*")")
    initial_duplx = Complex(strands=[incumbent, target],structure=intialDotParen)
    invader_cmplx = Complex(strands=[invader],structure=len(invader.sequence)*".", boltzmann_sample=True)
    incmbt_cmplx = Complex(strands=[incumbent],structure=len(incumbent.sequence)*".") 

    stop_success = StopCondition(Literals.success, [(incmbt_cmplx, Literals.dissoc_macrostate, 0)])
    stop_fail = StopCondition(Literals.failure, [(initial_duplx, Literals.dissoc_macrostate, 0)])

    opt.start_state = [initial_duplx, invader_cmplx]
    opt.stop_conditions = [stop_success, stop_fail]  

    return opt


def create_options_checkpoint(sim_no, traj_filename, idx, experiment_specs, simulation_specs):

    # Reconstruct options from a stored checkpoint 

    with h5.File(traj_filename, "r") as f:

        seed         = f[str(sim_no)]["checkpoint"+str(idx)]["seed"][()]
        strand_names = f[str(sim_no)]["checkpoint"+str(idx)]["strand_names"][()].decode() 
        structure    = f[str(sim_no)]["checkpoint"+str(idx)]["structure"][()].decode() 
        time    = f[str(sim_no)]["checkpoint"+str(idx)]["time"][()] 



    incumbent = Strand(name="incumbent", sequence=sequences["incumbent"])
    target    = Strand(name="target",    sequence=sequences["target"])
    invader   = Strand(name="invader",   sequence=sequences["invader"])

    initial_duplx = Complex(strands=[incumbent, target],
                            structure=len(incumbent.sequence)*"(" + 
                                      "+" + 
                                      (len(target.sequence)-len(incumbent.sequence))*"." + 
                                      len(incumbent.sequence)*")")


    incmbt_cmplx = Complex(strands=[incumbent],structure=len(incumbent.sequence)*".") 

    stop_success = StopCondition(Literals.success, [(incmbt_cmplx, Literals.dissoc_macrostate, 0)])
    stop_fail = StopCondition(Literals.failure, [(initial_duplx, Literals.dissoc_macrostate, 0)])

    opt = Options(
        simulation_mode = Literals.trajectory,
        **experiment_specs, 
        **simulation_specs)

    opt.DNA29Arrhenius()
    opt.simulation_start_time = time
    opt.simulation_time = opt.simulation_time + time

    opt.state_seed = seed

    L = [item.split(":")[-1] for item in strand_names.split(",")]
    strandmap = {"incumbent": incumbent,"target": target,"invader": invader}
    opt.start_state =  [Complex(strands=[strandmap[L[0]], strandmap[L[1]], strandmap[L[2]]],
                structure=structure)]

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



def pack_trajectory(opt: Options, timestep: float):

    strands = opt.strand_names()

    ordered_structs = []
    ordered_ids = []
    times = []
    energies = []    
    next_snapshot_time = timestep 

    for i, state in enumerate(opt.full_trajectory):
        current_time = opt.full_trajectory_times[i]
        # print("Current time: ", current_time)
        # print("Next snapshot at: ", next_snapshot_time)
        # print('\n')
        
        # Record state if it's time for a snapshot OR it's the first state OR it's the last state
        if_record = (i == 0) or (i == len(opt.full_trajectory_times)-1) or (current_time >= next_snapshot_time)
        
        if if_record:
            ids = [[strands[n].id for n in cmplx.strand_names.split(',')] for cmplx in state]
            order = np.argsort([min(s) for s in ids])  # inducing an ordering of the complexes 
            energy = sum(cmplx.energy for cmplx in state)
            
            normalized = [normalizeCyclicPermutation(ids[i], state[i].sequence, state[i].structure) for i in order]
            seq = [s[0] for s in normalized]
            sct = [s[1] for s in normalized]

            ordered_structs.append(sct)
            ordered_ids.append([ids[i] for i in order])
            energies.append(energy)
            times.append(current_time)
            
            # Update the next time to take a snapshot
            while next_snapshot_time <= current_time:
                next_snapshot_time += timestep
    
    structs = np.array(
        [' '.join(s) for s in ordered_structs],
        dtype=str)
    energies = np.array(energies, dtype=np.float64)
    times = np.array(times, dtype=np.float64)
    end = opt.interface.end_states
    
    print("Total number of collected states: ", len(structs))
    sys.stdout.flush()
    
    return (structs, energies, times, end, ordered_ids)



def incomplete(stoplog): 
    a = sorted([item.split(":")[-1] for item in stoplog[0][0].strand_names.split(",")])

    return a == ['incumbent', 'invader', 'target']


def simulate(opt: Options, timestep: float, verbose: bool=False):
    sys = SimSystem(opt)
    sys.start()
    if verbose:
        printTrajectory(opt, show_seed=True)
    
    return pack_trajectory(opt, timestep)


def batched_simulation(opt, sim_no, traj_filename, timestep):

    structs, energies, times, stoplog, ordered_ids = simulate(opt, timestep, verbose=False)

    save_sim(sim_no, traj_filename, structs, energies, times, opt.interface_trajectory_seed, ordered_ids)

    simulation_time = opt.simulation_time
    idx = len(structs)

    while incomplete(stoplog): 

        save_checkpoint(opt, sim_no, traj_filename, len(structs), idx)
        opt = bootstrap_options(opt, len(structs), times[-1]+simulation_time)

        structs, energies, times, stoplog, ordered_ids = simulate(opt, verbose=False)
        idx = append_sim(sim_no, traj_filename, structs, energies, times, ordered_ids)

    

def main():

    row = args.row
    nsims = args.nsims
    save_id = "" if args.save_id==None else "_"+args.save_id 
    timestep = args.timestep

    simulation_specs = {"num_simulations":1,
                        "output_interval":1,
                        "simulation_time":float('inf'),
                        }


    data_filename = "reactions/machinek/Machinek_SupTable6.csv"
    expid, sequences, experiment_specs = load_Machinek_row(data_filename, row)
    

    traj_filename = "reactions/machinek/" + expid +"/" + expid+save_id + ".hdf5"
    assert not Path(traj_filename).exists()
    
    for sim_no in range(nsims):
        print("\nStarting simulation", sim_no)
        sys.stdout.flush()
        
        opt = create_options_FSM(sequences, experiment_specs, simulation_specs)
        batched_simulation(opt, sim_no, traj_filename, timestep)
        
        


if __name__ == "__main__":
    main()


# Example: from the machinek directory run
    # python sample_trajs_machinek.py --row 0 --nsims 10 --save_id test --timestep 1e-7
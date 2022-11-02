from multistrand.objects import *
from multistrand.options import Options
from multistrand.system import SimSystem, energy
# save output to a text file
import sys 
import numpy as np


# preset
Loop_Energy = 0    # requesting no dG_assoc or dG_volume terms to be added.  So only loop energies remain.
Volume_Energy = 1  # requesting dG_volume but not dG_assoc terms to be added.  No clear interpretation for this.
Complex_Energy = 2 # requesting dG_assoc but not dG_volume terms to be added.  This is the NUPACK complex microstate energy, sans symmetry terms.
Tube_Energy = 3    # requesting both dG_assoc and dG_volume terms to be added.  Summed over complexes, this is the system state energy.

# Input different sequences and structures
SEQUENCE_TOT = ["ACUGAUCGUAGUCAC","AUUGAGCAUAUUCAC","CGGGCUAUUUAGCUG"] # I1,I2,I3
SEQUENCE = SEQUENCE_TOT[0] # I1
STRUCTURE = len(SEQUENCE) * '.'
NUM_SIM = 100000
ATIME_OUT = float('inf')  #0.000001
# ATIME_OUT = 0.0000001


# set multistrand simulator
c = Complex(strands=[Strand(name="hairpin",sequence=SEQUENCE)], structure=STRUCTURE)

o = Options(temperature=37, dangles='Some', start_state = [c], substrate_type="RNA",
            simulation_time = ATIME_OUT,  # 0.1 microseconds
            num_simulations = NUM_SIM,  # don't play it again, Sam
            output_interval = 1,  # record every single step
            rate_method = 'Metropolis', # the default is 'Kawasaki' (numerically, these are 1 and 2 respectively)
            rate_scaling = 'Calibrated', # this is the same as 'Default'.  'Unitary' gives values 1.0 to both.  
            simulation_mode = 'Trajectory',)  # numerically 128.  See interface/_options/constants.py for more info about all this.

# add stop conditions
success_complex = Complex(strands=[Strand(name="hairpin",sequence=SEQUENCE)], structure="..((((....)))).")
# stopSuccess = StopCondition("CLOSED", [(success_complex, 0, 0)])
stopSuccess = StopCondition("stop:FULL", [(success_complex, 0, 0)])
o.stop_conditions = [stopSuccess]

# simulate and print the trajectories
s = SimSystem(o)
s.start()

# save data to npy file
Struct=[]; Time=[]; Energy=[]
for i in range(len(o.full_trajectory)):
    time = o.full_trajectory_times[i]
    state = o.full_trajectory[i][0]
    struct = state[4]
    dG = state[5]

    Struct.append(struct)
    Time.append(time)
    Energy.append(dG)

with open('../output_data/I1_1M.npy', 'wb') as f:
    np.savez(f,structure=Struct,time=Time,energy=Energy)
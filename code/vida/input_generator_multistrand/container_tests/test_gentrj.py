from multistrand.objects import *
from multistrand.options import Options
from multistrand.system import SimSystem, energy
# save output to a text file
import sys 

stdoutOrigin=sys.stdout 
sys.stdout = open("../output_data/I1_10Ksim_test.txt", "w")

# preset
Loop_Energy = 0    # requesting no dG_assoc or dG_volume terms to be added.  So only loop energies remain.
Volume_Energy = 1  # requesting dG_volume but not dG_assoc terms to be added.  No clear interpretation for this.
Complex_Energy = 2 # requesting dG_assoc but not dG_volume terms to be added.  This is the NUPACK complex microstate energy, sans symmetry terms.
Tube_Energy = 3    # requesting both dG_assoc and dG_volume terms to be added.  Summed over complexes, this is the system state energy.

# define a function to print trajectories
def print_trajectory(o):
    # print o.full_trajectory[0][0][3]   # the strand sequence
    # print o.start_state[0].structure   # the starting structure
    for i in range(len(o.full_trajectory)):
        time = o.full_trajectory_times[i]
        state = o.full_trajectory[i][0]
        struct = state[4]
        dG = state[5]
        print struct + ' t=%11.9f seconds, dG=%6.2f kcal/mol' % (time, dG)

# Input different sequences and structures
SEQUENCE_TOT = ["ACUGAUCGUAGUCAC","AUUGAGCAUAUUCAC","CGGGCUAUUUAGCUG"] # I1,I2,I3
SEQUENCE = SEQUENCE_TOT[0] # I1
STRUCTURE = len(SEQUENCE) * '.'
NUM_SIM = 1
ATIME_OUT = float('inf')  #0.000001


# set multistrand simulator
c = Complex(strands=[Strand(name="hairpin",sequence=SEQUENCE)], structure=STRUCTURE)

o = Options(temperature=37, dangles='Some', start_state = [c], substrate_type="RNA",
            simulation_time = ATIME_OUT,  # 0.1 microseconds
            num_simulations = NUM_SIM,  # don't play it again, Sam
            output_interval = 1,  # record every single step
            rate_method = 'Metropolis', # the default is 'Kawasaki' (numerically, these are 1 and 2 respectively)
            rate_scaling = 'Calibrated', # this is the same as 'Default'.  'Unitary' gives values 1.0 to both.  
            simulation_mode = 'Trajectory',)
            # sodium=0.5,
            # magnesium=0)  # numerically 128.  See interface/_options/constants.py for more info about all this.

# add stop conditions
success_complex = Complex(strands=[Strand(name="hairpin",sequence=SEQUENCE)], structure="..((((....)))).")
# stopSuccess = StopCondition("CLOSED", [(success_complex, 0, 0)])
stopSuccess = StopCondition("stop:FULL", [(success_complex, 0, 0)])
o.stop_conditions = [stopSuccess]


# print "k_uni = %g /s, k_bi = %g /M/s" % (o.unimolecular_scaling, o.bimolecular_scaling)  # you can also set them to other values if you want

# simulate and print the trajectories
s = SimSystem(o)
s.start()
print_trajectory(o)


# save output to a text file
sys.stdout.close()
sys.stdout=stdoutOrigin




"""
##### Not Work for plot ######
##############################

# plot save check
import matplotlib.pyplot as plt
# creating data and plotting a histogram
x =[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
plt.hist(x)
  
# saving the figure.
plt.savefig("squares1.png",
            bbox_inches ="tight",
            pad_inches = 1,
            transparent = True,
            facecolor ="g",
            edgecolor ='w',
            orientation ='landscape')
# fig.savefig("plt_test")
"""

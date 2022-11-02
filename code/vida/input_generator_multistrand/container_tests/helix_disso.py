from multistrand.objects import Complex, Domain, Strand, StopCondition
from multistrand.options import Options, Literals
from multistrand.system import SimSystem

# save output to a text file
import sys 
stdoutOrigin=sys.stdout 
sys.stdout = open("../output_data/disso_PT3.txt", "w")


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
mySeq = "TGACGATCATGTCTGCGTGACTAGA"
NUM_SIM = 1
ATIME_OUT = float('inf')  #0.000001


# Using domain representation makes it easier to write secondary structures.
onedomain = Domain(name="itall", sequence=mySeq)
top = Strand(name="top", domains=[onedomain])
bot = top.C

# Note that the structure is specified to be single stranded, but this will be over-ridden when Boltzmann sampling is turned on.
startTop = Complex(strands=[top], structure=".")
startBot = Complex(strands=[bot], structure=".")

# set multistrand options
o = Options(temperature=20, dangles='Some', start_state = [startTop, startBot], 
            substrate_type="DNA",
            simulation_time = ATIME_OUT,  # 0.1 microseconds
            num_simulations = NUM_SIM,  # don't play it again, Sam
            output_interval = 1,  # record every single step
            rate_method = 'Metropolis', # the default is 'Kawasaki' (numerically, these are 1 and 2 respectively)
            rate_scaling = 'Calibrated', # this is the same as 'Default'.  'Unitary' gives values 1.0 to both.  
            simulation_mode = 'Trajectory',)
            # sodium=0.5,
            # magnesium=0)  # numerically 128.  See interface/_options/constants.py for more info about all this.

# add stop conditions

# Note that "+" is used to indicate strand breaks.
# Stop when the exact full duplex is achieved.
success_complex = Complex(strands=[top, bot], structure="(+)")
stopSuccess = StopCondition(Literals.success, [(success_complex, Literals.exact_macrostate, 0)])

# Declare the simulation unproductive if the strands become single-stranded again.
failed_complex = Complex(strands=[top], structure=".")
stopFailed = StopCondition(Literals.failure, [(failed_complex, Literals.dissoc_macrostate, 0)])

# o.stop_conditions = [stopSuccess,stopFailed]
o.stop_conditions = [stopSuccess]

# simulate and print the trajectories
s = SimSystem(o)
s.start()
print_trajectory(o)


# save output to a text file
sys.stdout.close()
sys.stdout=stdoutOrigin


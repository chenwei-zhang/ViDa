from multistrand.objects import Complex, Domain, Strand, StopCondition
from multistrand.options import Options, Literals
from multistrand.system import SimSystem
import sys 
import os
import numpy as np

# define a function to print trajectories
def print_trajectory(o):
    # print o.full_trajectory[0][0][3]   # the strand sequence
    # print o.start_state[0].structure   # the starting structure
    for i in range(len(o.full_trajectory)):
        time = o.full_trajectory_times[i]
        state = o.full_trajectory[i][0]
        struct = state[4]
        dG = state[5]
        print(struct + ' t=%11.9f seconds, dG=%6.2f kcal/mol' % (time, dG))

# define helix association function
def association(mySeq,Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL,Prt):
    # Using domain representation makes it easier to write secondary structures.
    onedomain = Domain(name="itall", sequence=mySeq)
    top = Strand(name="top", domains=[onedomain])
    bot = top.C

    # Note that the structure is specified to be single stranded, but this will 
    # be over-ridden when Boltzmann sampling is turned on.
    startTop = Complex(strands=[top], structure=".")
    startBot = Complex(strands=[bot], structure=".")

    # set multistrand options
    o = Options(temperature=Temp, dangles='Some', start_state = [startTop, startBot], 
                substrate_type=substrate_type,parameter_type="Nupack",
                simulation_time = ATIME_OUT,  # 0.1 microseconds
                num_simulations = NUM_SIM,  # don't play it again, Sam
                output_interval = 1,  # record every single step
                rate_method = 'Metropolis', # the default is 'Kawasaki' (numerically, these are 1 and 2 respectively)
                rate_scaling = 'Calibrated', # this is the same as 'Default'.  'Unitary' gives values 1.0 to both.  
                simulation_mode = 'Trajectory',)
                # sodium=0.5, # default=1.0
                # magnesium=0)  # default=0.0

    """ # add stop conditions
    # for StopCondition and Macrostate definitions:
    Exact_Macrostate = 0   # match a secondary structure exactly (i.e. any system state that has a complex with this exact structure)
    Bound_Macrostate = 1   # match any system state in which the given strand is bound to another strand
    Dissoc_Macrostate = 2  # match any system state in which there exists a complex with exactly the given strands, in that order
    Loose_Macrostate = 3   # match a secondary structure with "don't care"s, allowing a certain number of disagreements
    Count_Macrostate = 4   # match a secondary structure, allowing a certain number of disagreements
    # see Schaeffer's PhD thesis, chapter 7.2, for more information
    """
    # Stop when the exact full duplex is achieved.
    success_complex = Complex(strands=[top, bot], structure="(+)")
    stopSuccess = StopCondition(Literals.success, [(success_complex, Literals.exact_macrostate, 0)])
    # Declare the simulation unproductive if the strands become single-stranded again.
    failed_complex = Complex(strands=[top], structure=".")
    stopFailed = StopCondition(Literals.failure, [(failed_complex, Literals.dissoc_macrostate, 0)])
    
    if FAIL == "ON":
        o.stop_conditions = [stopSuccess,stopFailed]
    elif FAIL == "OFF": 
        o.stop_conditions = [stopSuccess]
    
    # start simulatiomn
    s = SimSystem(o)
    s.start()

    # if Prt == "SUCC" and o.interface.results[0].tag == "SUCCESS":
    #     print_trajectory(o)  # only print successful trajectories
    # elif Prt == "ALL": 
    #     print_trajectory(o)  # print all trajectories

    return o


# Input arguments
mySeq = "TGACGATCATGTCTGCGTGACTAGA"  # PT3
# mySeq = "GAGACTTGCCATCGTAGAACTGTTG"  # PT0
Temp = 20
substrate_type = "DNA"
NUM_SIM = 1
ATIME_OUT = float('inf')  #0.000001


"""
Run simulator once (num_simulations>=1)
"""
# save output to a text file #
def sim_num(NUM_SIM=NUM_SIM,ATIME_OUT=ATIME_OUT,Temp=Temp,
    substrate_type=substrate_type,mySeq=mySeq,FAIL="ON",Prt="SUCC"):
    stdoutOrigin=sys.stdout 
    sys.stdout = open("../output_data/helix_asso_succ_files_test/assos_PT3_{}sim_{}C_simnum.txt".format(NUM_SIM,Temp), "w")

    # running simulation
    o = association(mySeq,Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL,Prt)

    # save output to a text file #
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    
    return o


"""
Run simulator many times (num_simulations=1)
"""
# # save output to a text file #
# def sim_loop(N,NUM_SIM=NUM_SIM,ATIME_OUT=ATIME_OUT,Temp=Temp,
#     substrate_type=substrate_type,mySeq=mySeq,FAIL="ON",Prt="SUCC"):
#     for i in range(N):
#         # # save output to a text file #
#         stdoutOrigin=sys.stdout 
#         sys.stdout = open("../output_data/helix_asso_succ_files_test/assos_PT0_{}sim_{}C_{}.txt".format(NUM_SIM,Temp,i), "w")
        
#         # running simulation
#         o = association(mySeq,Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL,Prt)
        
#         # save output to a text file #
#         sys.stdout.close()
#         sys.stdout=stdoutOrigin

#         # remove files below size 
#         file = "../output_data/helix_asso_succ_files_test/assos_PT0_{}sim_{}C_{}.txt".format(NUM_SIM,Temp,i)
#         # if os.path.getsize(file) < 110:
#         if o.interface.results[0].tag == "SUCCESS":
#             os.remove(file)
#     return o


# save output to a text file #
def sim_loop(N,NUM_SIM=NUM_SIM,ATIME_OUT=ATIME_OUT,Temp=Temp,
    substrate_type=substrate_type,mySeq=mySeq,FAIL="ON",Prt="SUCC"):
    
    i = 0
    while i < N:
        # running simulation
        o = association(mySeq,Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL,Prt)
        
        if Prt == "SUCC" and o.interface.results[0].tag == "SUCCESS":
            # save output to a text file #
            stdoutOrigin=sys.stdout 
            sys.stdout = open("../output_data/helix_assos_PT3/assos_PT3_{}sim_{}C_{}.txt".format(NUM_SIM,Temp,i), "w")
        
            print_trajectory(o)

            # save output to a text file #
            sys.stdout.close()
            sys.stdout=stdoutOrigin

            i+=1
            
    return o


""" NOTE:
### If FAIL="OFF", it will print all trajectories,
### as all are successful, then no matter Prt setting
"""
# # Run num_sim
# o = sim_num(NUM_SIM=10)
# o = sim_num(NUM_SIM=30,FAIL="ON",Prt="ALL")

# Run sim loops
o = sim_loop(N=100)
## o = sim_loop(N=3,FAIL="ON",Prt="ALL")


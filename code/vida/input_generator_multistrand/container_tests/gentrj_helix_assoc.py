from multistrand.objects import Complex, Domain, Strand, StopCondition
from multistrand.options import Options, Literals
from multistrand.system import SimSystem
import sys 
import os
import numpy as np


# define a function to print trajectories
def print_trajectory(o):
    """ print_trajectory file meaning:
    (random number seed, unique complex id, strand names, sequence, structure, energy )
    """
    # print o.full_trajectory[0][0][3]   # the strand sequence
    # print o.start_state[0].structure   # the starting structure
    for i in range(len(o.full_trajectory)):
        # time = o.full_trajectory_times[i]
        # state = o.full_trajectory[i][0]
        # struct = state[4]
        # dG = state[5]
        all = o.full_trajectory[i]
        # print(struct + ' t=%11.9f seconds, dG=%6.2f kcal/mol' % (time, dG))
        print all


# define a function to print trajectories of helix association
def print_trajectory_association(o,n):
    op = o.full_trajectory
    for i in range(len(op)):
        time = o.full_trajectory_times[i]

        if len(op[i]) == 2:
            if op[i][0][2]=='{}:top*'.format(n) and op[i][1][2]=='{}:top'.format(n):
                struct = op[i][1][4] + "+" + op[i][0][4]
                dG = op[i][1][5] + op[i][0][5]

                print(struct + ' t={} seconds, dG={} kcal/mol, ' + "0").format(time, dG)
            else:
                raise ValueError('Disordered output structures appeared len=2.')

        elif len(op[i]) == 1:
            state = op[i][0]
            if state[2] == '{}:top,{}:top*'.format(n,n):
                struct = state[4]
            elif state[2] == '{}:top*,{}:top'.format(n,n):
                ss0 = state[4].split("+")[0]
                ss1 = state[4].split("+")[1]

                previous_state0 = op[i-1][0][4]
                previous_state1 = op[i-1][1][4]
            
                for j in range(len(ss0)):
                    if ss0[j] != previous_state0[j]:
                        if ss0[j] != "(" or previous_state0[j] != ".":
                            raise ValueError('Wrong function.')

                        list0 = list(ss0)
                        list0[j] = ")"
                        ss0 = ''.join(list0)

                    if ss1[j] != previous_state1[j]:
                        if ss1[j] != ")" or previous_state1[j] != ".":
                            raise ValueError('Wrong function.')

                        list1 = list(ss1)
                        list1[j] = "("
                        ss1 = ''.join(list1)

                struct = ss1 + "+" + ss0

            else:
                raise ValueError('Disordered output structures appeared.')
            dG = state[5]

            print(struct + ' t={} seconds, dG={} kcal/mol, ' + "1").format(time, dG)

        else:
            raise ValueError('Not a valid helix assosiation output.')


# define PT0 helix association function
def association_PT0(Seq,Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL):

    # Using domain representation makes it easier to write secondary structures.
    onedomain = Domain(name="itall", sequence=Seq)
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
                simulation_mode = 'Trajectory',
                sodium=0.5, # default=1.0
                # magnesium=0)  # default=0.0
                )
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

    return o


# define PT3 helix association function
def association_PT3(Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL):
    PT3 = "TGACGATCATGTCTGCGTGACTAGA"  # PT3

    # Using domain representation makes it easier to write secondary structures.
    onedomain = Domain(name="itall", sequence=PT3)
    top = Strand(name="top", domains=[onedomain])
    bot = top.C

    top_dp = "."*10+"("*3+"."*5+")"*3+"."*4  # top_P3: 11,12,13 --- 19,20,21
    bot_dp = "."*4+"("*3+"."*5+")"*3+"."*10  # bot_T3: 5,6,7 --- 13,14,15

    # Note that the structure is specified to be single stranded, but this will 
    # be over-ridden when Boltzmann sampling is turned on.
    startTop = Complex(strands=[top], structure=top_dp)
    startBot = Complex(strands=[bot], structure=bot_dp)

    # set multistrand options
    o = Options(temperature=Temp, dangles='Some', start_state = [startTop, startBot], 
                substrate_type=substrate_type,parameter_type="Nupack",
                simulation_time = ATIME_OUT,  # 0.1 microseconds
                num_simulations = NUM_SIM,  # don't play it again, Sam
                output_interval = 1,  # record every single step
                rate_method = 'Metropolis', # the default is 'Kawasaki' (numerically, these are 1 and 2 respectively)
                rate_scaling = 'Calibrated', # this is the same as 'Default'.  'Unitary' gives values 1.0 to both.  
                simulation_mode = 'Trajectory',
                sodium=0.5, # default=1.0
                # magnesium=0  # default=0.0
                )
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

    return o


# define PT4 helix association function
def association_PT4(Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL):
    PT4 = "ACACGATCATGTCTGCGTGACTAGA"  # PT4

    # Using domain representation makes it easier to write secondary structures.
    onedomain = Domain(name="itall", sequence=PT4)
    top = Strand(name="top", domains=[onedomain])
    bot = top.C

    top_dp = "."*1+"("*4+"."*10+")"*4+"."*6  # top_P4: 2,3,4,5 --- 16,17,18,19
    bot_dp = "."*6+"("*4+"."*10+")"*4+"."*1  # bot_T4: 7,8,9,10 --- 21,22,23,24

    # Note that the structure is specified to be single stranded, but this will 
    # be over-ridden when Boltzmann sampling is turned on.
    startTop = Complex(strands=[top], structure=top_dp)
    startBot = Complex(strands=[bot], structure=bot_dp)

    # set multistrand options
    o = Options(temperature=Temp, dangles='Some', start_state = [startTop, startBot], 
                substrate_type=substrate_type,parameter_type="Nupack",
                simulation_time = ATIME_OUT,  # 0.1 microseconds
                num_simulations = NUM_SIM,  # don't play it again, Sam
                output_interval = 1,  # record every single step
                rate_method = 'Metropolis', # the default is 'Kawasaki' (numerically, these are 1 and 2 respectively)
                rate_scaling = 'Calibrated', # this is the same as 'Default'.  'Unitary' gives values 1.0 to both.  
                simulation_mode = 'Trajectory',
                sodium=0.5, # default=1.0
                # magnesium=0  # default=0.0
                )
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

    return o


Temp = 20
substrate_type = "DNA"
ATIME_OUT = float('inf')  #0.000001

"""
Run simulator many times
"""
# save output to a text file #
def sim_loop(seq,N,NUM_SIM=1,ATIME_OUT=ATIME_OUT,Temp=Temp,
    substrate_type=substrate_type,FAIL="OFF",og="False"):
    
    n = 0
    while n < N:
        # running simulation
        if seq == "PT0":
            Seq = "GAGACTTGCCATCGTAGAACTGTTG"  # PT0
            o = association_PT0(Seq,Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL)
        elif seq == "PT3":
            Seq = "TGACGATCATGTCTGCGTGACTAGA"  # PT3
            o = association_PT0(Seq,Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL)
        elif seq == "PT4":
            Seq = "ACACGATCATGTCTGCGTGACTAGA"  # PT4
            o = association_PT0(Seq,Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL)
        elif seq == "PT3_hairpin":
            o = association_PT3(Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL)
        elif seq == "PT4_hairpin": 
            o = association_PT4(Temp,substrate_type,NUM_SIM,ATIME_OUT,FAIL)
            
        stdoutOrigin=sys.stdout 
        # sys.stdout = open("../output_data/helix_assoc_PT0/assoc_PT0_{}sim_{}C_{}.txt".format(NUM_SIM,Temp,n), "w")
        sys.stdout = open("../output_data/assoc_{}_{}sim_{}C_{}.txt".format(seq,NUM_SIM,Temp,n), "w")

        if og == False:
            print_trajectory_association(o,n)
        elif og == True:
            print_trajectory(o)

        # save output to a text file #
        sys.stdout.close()
        sys.stdout=stdoutOrigin
        
        n+=1
    return o

###########################################
# # generate 100 trajectories #
###########################################
# seq = "PT4_hairpin"
seq = "PT0"
o = sim_loop(seq,N=1,og=True)  # Run sim loops


# ############################################
# # run 10000 times to collect reaction time #
# ############################################
# # PT0
# mySeq = "GAGACTTGCCATCGTAGAACTGTTG"  # PT0
# TimePT0=np.array([])
# for i in range(10000):
#     o = association(mySeq,Temp,substrate_type,NUM_SIM=1,ATIME_OUT=ATIME_OUT,FAIL="OFF")
#     time = o.full_trajectory_times
#     TimePT0 = np.append(TimePT0, time)
# # PT3
# mySeq = "TGACGATCATGTCTGCGTGACTAGA"  # PT3
# TimePT3=np.array([])
# for i in range(10000):
#     o = association(mySeq,Temp,substrate_type,NUM_SIM=1,ATIME_OUT=ATIME_OUT,FAIL="OFF")
#     time = o.full_trajectory_times
#     TimePT3 = np.append(TimePT3, time)

# f_data = "../output_data/helix_assoc_time.npz"
# with open(f_data, 'wb') as f:
#     np.savez(f_data,
#             PT0_time=TimePT0, PT3_time=TimePT3)

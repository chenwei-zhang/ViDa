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
        print (all)


# define a function to print trajectories of helix association
def print_trajectory_zhang2007(o,n):
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


# define the reaction function for zhang2007 paper
def zhang2007(ATIME_OUT=float('inf')):
    # Define Fig.2 sequences and their structures
    # OB (Output)
    OB1 = "CTTTCCTACA"
    OB2 = "CCTACGTCTCCAACTAACTTACGG"
    OB = OB1 + OB2
    # F (Fuel)
    F2 = "CCTACGTCTCCAACTAACTTACGG"
    F3 = "CCCT"
    F4 = "CATTCAATACCCTACG" 
    F = F2 + F3 + F4
    # C (Catalyst)
    C4 = "CATTCAATACCCTACG"
    C5 = "TCTCCA"
    C = C4 + C5
    # substrate
    LB = "TGGAGACGTAGGGTATTGAATGAGGGCCGTAAGTTAGTTGGAGACGTAGG" # I4_bot, sustrate
    # I4 dot-parenthesis notation
    I4_dp = "."*len(OB1)+"("*len(OB2)+ "+" + "."*len(F2)+"("*len(F3)+"."*len(F4) + "+" + ")"*(len(OB2)+len(F3))+"("*len(C) + "+" + ")"*len(C)
    # define domain
    dm_OB = Domain(name="OB",sequence=OB,length=len(OB))
    dm_F = Domain(name="F",sequence=F,length=len(F))
    dm_LB = Domain(name="LB",sequence=LB,length=len(LB))
    dm_C = Domain(name="C",sequence=C,length=len(C))

    # define start Complex I4
    start_complex = Complex(strands=[dm_OB, dm_F, dm_LB, dm_C], structure=I4_dp)

    # set multistrand options
    o = Options(temperature=25, dangles='Some', start_state = [start_complex], 
                substrate_type="DNA",parameter_type="Nupack",
                simulation_time = ATIME_OUT,  # float('inf') infinite
                num_simulations = 1,  # don't play it again, Sam
                output_interval = 1,  # record every single step
                rate_method = 'Metropolis', # the default is 'Kawasaki' (numerically, these are 1 and 2 respectively)
                rate_scaling = 'Calibrated', # this is the same as 'Default'.  'Unitary' gives values 1.0 to both.  
                simulation_mode = 'Trajectory',)
                # sodium=0.5, # default=1.0
                # magnesium=0)  # default=0.0

    # success strands setting:
    success_W_dp = "("*len(F) + "+" + ")"*len(F)+"."*len(C5)
    success_C_dp = "."*len(C)
    success_OB_dp = "."*len(OB)

    # Stop when the exact full duplex is achieved.
    success_complex_W = Complex(strands=[dm_F,dm_LB], structure=success_W_dp)
    success_complex_C = Complex(strands=[dm_C], structure=success_C_dp)
    success_complex_OB= Complex(strands=[dm_OB], structure=success_OB_dp)


    stopSuccess_W = StopCondition(Literals.success, [(success_complex_W, Literals.exact_macrostate, 0)])
    stopSuccess_C = StopCondition(Literals.success, [(success_complex_C, Literals.exact_macrostate, 0)])
    stopSuccess_OB = StopCondition(Literals.success, [(success_complex_OB, Literals.exact_macrostate, 0)])

    # o.stop_conditions = [stopSuccess_W,stopSuccess_C,stopSuccess_OB]
    # o.stop_conditions = [stopSuccess_W]
    o.stop_conditions = [stopSuccess_W,stopSuccess_C]

    

    # start simulatiomn
    s = SimSystem(o)
    s.start()

    return o


"""
Run simulator many times
"""
# save output to a text file #
def sim_loop(N):
    n = 0
    while n < N:
        # running simulation
        o = zhang2007()

        stdoutOrigin=sys.stdout 
        # sys.stdout = open("../output_data/helix_assoc_PT3_new_2/assoc_PT3_{}sim_{}C_{}.txt".format(NUM_SIM,Temp,n), "w")
        sys.stdout = open("../output_data/zhang2007_25C_{}.txt".format(n), "w")

        print_trajectory_zhang2007(o,n)

        # save output to a text file #
        sys.stdout.close()
        sys.stdout=stdoutOrigin
        
        n+=1
    return o



# Run sim loops
# o = sim_loop(N=100)


o = zhang2007(ATIME_OUT=float('inf'))


stdoutOrigin=sys.stdout 
sys.stdout = open("../output_data/zhang2007_25C_og.txt", "w")
print_trajectory(o)
# save output to a text file #
sys.stdout.close()
sys.stdout=stdoutOrigin
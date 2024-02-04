import sys
from multistrand.objects import *
from multistrand.options import Options, EnergyType, Literals
from multistrand.system import SimSystem, calculate_energy


def print_trajectory(o):
    seqstring=''
    for i in range(len(o.full_trajectory)): # go through each output microstate of the trajectory
        time = o.full_trajectory_times[i]   # time at which this microstate is entered
        states = o.full_trajectory[i]       # this is a list of the complexes present in this tube microstate
        newseqs = []
        for state in states: newseqs += [ state[3] ]   # extract the strand sequences in each complex (joined by "+" for multistranded complexes)
        newseqstring = ' '.join(newseqs)    # make a space-separated string of complexes, to represent the whole tube system sequence
        if not newseqstring == seqstring :
            print(newseqstring)
            seqstring=newseqstring          # because strand order can change upon association of dissociation, print it when it changes
        structs = []
        for state in states: structs += [ state[4] ]   # similarly extract the secondary structures for each complex
        tubestruct = ' '.join(structs)      # give the dot-paren secondary structure for the whole test tube
        dG=0
        for state in states: dG += state[5]
        print('%s t=%11.9f seconds, dG=%6.2f kcal/mol' % (tubestruct,time, dG))

        # Needlessly verify that the reported trajectory energies are the Tube_Energy values
        dGv=0
        for state in states:
            cs=state[3].split('+')
            st=state[4]
            dGv += calculate_energy( [Complex( strands=[Strand(sequence=s) for s in cs], structure=st)], o, EnergyType.tube)[0]
        if not dGv == dG: print("Energy Mismatch")
        
        

def create_setup():
    toehold_length = 7
    toehold_seq = 'ATGTGGAGGG'
    bm_design = "GGTGAGTTTGAGGTTGA"
    incum_extra_design = "TGGTGTTTGTGGGTGT"

    # creating Domain objects
    toehold = Domain(name="toehold", sequence=toehold_seq[0:toehold_length])
    branch_migration = Domain(name="bm", sequence=bm_design)

    # invader
    invader = branch_migration + toehold
    invader.name = "invader"

    # target (substrate)
    toehold_extra = Domain(name="toehold_extra", sequence=toehold_seq[toehold_length:]) 
    target = toehold_extra.C + toehold.C + branch_migration.C
    target.name = "target"

    # incumbent
    incumbent_extra = Domain(name="incumbent_extra", sequence=incum_extra_design)
    incumbent = incumbent_extra+branch_migration
    incumbent.name = "incumbent"

    # creates the target-incumbent start complex  
    start_complex_substrate_incumbent = Complex(strands=[target, incumbent], structure="..(+.)")
    # creates invader complex. 
    start_complex_incoming = Complex(strands=[invader], structure="..") 

    # creates a complex for a "succcessful displacement" stop condition. This is the incumbent strand forming a complex of its own which means it has been displaced.
    complete_complex_success = Complex(strands=[incumbent], structure="..")
    success_stop_condition = StopCondition("SUCCESS", [(complete_complex_success, Literals.dissoc_macrostate, 0)])
    

    # # complex to create failed displacement stop condition; incumbent falls off.   
    # failed_complex = Complex(strands=[invader], structure="..")  
    # failed_stop_con`ditions = StopCondition("FAILURE", [(failed_complex, Literals.dissoc_macrostate, 0)]) 
        
        
    o = Options(
        simulation_mode="Trajectory",   #  First Step
        substrate_type="DNA",
        num_simulations=1, 
        simulation_time=float('inf'),  # note the float('inf') second simulation time, to make sure simulations finish
        dangles="Some", 
        temperature=4, 
        join_concentration = 100e-6, # 100 uM
        gt_enable = False,
        output_interval = 1, # record every 1000 steps
        verbosity=0,
        start_state = [start_complex_incoming, start_complex_substrate_incumbent],
        # stop_conditions = [success_stop_condition, failed_stop_condition]
        stop_conditions = [success_stop_condition]
        )
    
    o.DNA23Metropolis()

    return o


if __name__ == '__main__':
    
    for i in range(100):
            
        o = create_setup()
        s = SimSystem(o)
        s.start()

        stdoutOrigin=sys.stdout         
        
        sys.stdout = open(f"./raw_data/Machinek-PRF/Machinek-PRF-{i}.txt", "w")
        print_trajectory(o)
        sys.stdout.flush()  # Flush the output here
        sys.stdout.close()
        sys.stdout=stdoutOrigin
        
        print(f"Simulation {i} complete; Trajecory length: {len(o.full_trajectory)}")
        sys.stdout.flush()  # Flush the output here as well